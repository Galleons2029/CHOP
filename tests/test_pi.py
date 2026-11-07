import pytest
import torch

from chop.pi import controls
from chop.pi.composition import BASE, PIConditionalTeacher, compose_logits, left_pad
from chop.pi.sources import PROTOCOLS, is_deployable, protocol_sources


def test_protocols():
    assert is_deployable(PROTOCOLS["swe_deployable"])
    assert not is_deployable(PROTOCOLS["swe_oracle"])
    assert [s.name for s in protocol_sources("math")] == ["final_answer"]
    with pytest.raises(KeyError):
        protocol_sources(["nope"])


def test_compose_logits(gen):
    base = torch.randn(4, 7, generator=gen)
    a, b = torch.randn(4, 7, generator=gen), torch.randn(4, 7, generator=gen)
    assert torch.allclose(compose_logits(base, [a]), a)
    assert torch.allclose(compose_logits(base, [a, b]), a + b - base)


def test_left_pad_positions():
    ids, mask, pos = left_pad([[5, 6, 7], [8]], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [0, 0, 8]]
    assert pos.tolist() == [[0, 1, 2], [0, 0, 0]]


def _peft_teacher(gen):
    peft = pytest.importorskip("peft")
    from chop.tiny import tiny_qwen3

    base = tiny_qwen3()
    cfg = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"], init_lora_weights=False)
    model = peft.get_peft_model(base, cfg, adapter_name="call_graph")
    model.add_adapter("test_manifest", cfg)
    model.eval()
    return model


def _single(model, name, ids):
    ids = torch.tensor([ids])
    if name == BASE:
        with model.disable_adapter():
            return model(input_ids=ids).logits[0]
    model.set_adapter(name)
    return model(input_ids=ids).logits[0]


def test_score_matches_individual_forwards(gen):
    model = _peft_teacher(gen)
    teacher = PIConditionalTeacher(model, pad_id=0)
    prompt, resp = [5, 6, 7, 8], [9, 10, 11]
    prefix = {"call_graph": [20, 21, 22, 23, 24], "test_manifest": [30, 31]}
    base, comp = teacher.score(prompt, resp, prefix, ["call_graph", "test_manifest"])
    with torch.no_grad():
        b = _single(model, BASE, prompt + resp)[3:6]
        cg = _single(model, "call_graph", prefix["call_graph"] + prompt + resp)[8:11]
        tm = _single(model, "test_manifest", prefix["test_manifest"] + prompt + resp)[5:8]
    assert torch.allclose(base, b, atol=1e-5)
    assert torch.allclose(comp, cg + tm - b, atol=1e-5)
    base_only, none = teacher.score(prompt, resp, prefix, [])
    assert none is None and torch.allclose(base_only, b, atol=1e-5)


def test_prompt_interface_uses_base_model(gen):
    model = _peft_teacher(gen)
    teacher = PIConditionalTeacher(model, pad_id=0, interface="prompt")
    prompt, resp = [5, 6], [9, 10]
    prefix = {"call_graph": [20, 21], "test_manifest": [30]}
    _, comp = teacher.score(prompt, resp, prefix, ["call_graph", "test_manifest"])
    with torch.no_grad():
        ref = _single(model, BASE, [20, 21, 30] + prompt + resp)[4:6]
    assert torch.allclose(comp, ref, atol=1e-5)


def test_composite_sampling_greedy_matches_manual(gen):
    model = _peft_teacher(gen)
    teacher = PIConditionalTeacher(model, pad_id=0)
    ctx = [5, 6, 7]
    prefix = {"call_graph": [20, 21, 22], "test_manifest": [30]}
    toks, lens = teacher.sample(ctx, prefix, ["call_graph", "test_manifest"], num_samples=2,
                                max_new_tokens=4, temperature=1e-6, generator=gen)
    assert toks.shape == (2, 4) and lens.tolist() == [4, 4]
    seq = list(ctx)
    with torch.no_grad():
        for _ in range(4):
            b = _single(model, BASE, seq)[-1]
            cg = _single(model, "call_graph", prefix["call_graph"] + seq)[-1]
            tm = _single(model, "test_manifest", prefix["test_manifest"] + seq)[-1]
            seq.append(int(compose_logits(b, [cg, tm]).argmax()))
    assert toks[0].tolist() == seq[3:] and toks[1].tolist() == seq[3:]


def test_content_controls(gen):
    blocks = ["a", "b", "c", "d"]
    shuffled = controls.shuffle_within_batch(blocks, gen)
    assert sorted(shuffled) == blocks and all(x != y for x, y in zip(shuffled, blocks, strict=True))
    repos = ["r1", "r1", "r2", "r2"]
    wrong = controls.wrong_repo(blocks, repos, gen)
    for i, w in enumerate(wrong):
        assert repos[blocks.index(w)] != repos[i]
    assert controls.apply_content_control("sentinel", blocks) == ["[NO_PI]"] * 4


def test_randomize_lora_preserves_norms(gen):
    model = _peft_teacher(gen)
    before = {n: p.norm().item() for n, p in model.named_parameters() if ".call_graph." in n and "lora_" in n}
    k = controls.randomize_lora_(model, "call_graph", gen)
    after = {n: p.norm().item() for n, p in model.named_parameters() if ".call_graph." in n and "lora_" in n}
    assert k == len(before) > 0
    for n in before:
        assert abs(before[n] - after[n]) < 1e-4
