import random

import pytest
import torch

from adapters import calibration, triples
from adapters.train_lora import add_lora, encode_triple, train
from envs.toy import ToyTokenizer

PATCH = "diff --git a/pkg/core.py b/pkg/core.py\n@@ -10,3 +10,4 @@\n"


def test_expert_alignment_by_hunk_overlap():
    steps = [triples.Step("s0", "view", []),
             triples.Step("s1", "edit core", [triples.Edit("pkg/core.py", 11, 12)]),
             triples.Step("s2", "edit other", [triples.Edit("pkg/other.py", 11, 12)]),
             triples.Step("s3", "edit far", [triples.Edit("pkg/core.py", 50, 60)])]
    assert [i for i, _ in triples.expert_steps(steps, PATCH)] == [1]


def test_swe_triples_pair_every_source():
    inst = {"instance_id": "i", "patch": PATCH, "FAIL_TO_PASS": "[\"t::a\"]", "PASS_TO_PASS": "[]"}
    steps = [triples.Step("s1", "edit", [triples.Edit("pkg/core.py", 10, 11)])]
    out = triples.swe_triples(inst, steps, ["test_manifest", "next_file", "call_graph"], None, "g", random.Random(0))
    assert {t.source for t in out} == {"test_manifest", "next_file", "call_graph"}
    assert all(t.action == "edit" for t in out)


def test_encode_triple_masks_all_but_action():
    tok = ToyTokenizer()
    e = encode_triple(tok, {"source": "hint", "pi": "42", "state": "Q:1+1", "action": "A:2"}, max_len=200)
    n_action = len(tok.encode("A:2")) + 1
    assert all(x == -100 for x in e["labels"][:-n_action]) and e["labels"][-n_action:] == e["input_ids"][-n_action:]


def test_lora_training_reduces_loss():
    pytest.importorskip("peft")
    from chop.tiny import tiny_qwen3

    tok = ToyTokenizer()
    model = add_lora(tiny_qwen3(vocab_size=128), "hint")
    data = [{"source": "hint", "pi": "7", "state": f"Q:{i}", "action": "A:7"} for i in range(8)]
    losses = train(model, tok, data, None, epochs=30, lr=5e-3, batch_size=8, grad_accum=1, warmup=1,
                   log=lambda _: None)
    assert losses[-1] < 0.9 * losses[0]      # tiny frozen-head model: capacity-limited, but it must learn
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)


def test_calibration_diagnostics(gen):
    a = torch.randn(10, 7, generator=gen)
    assert torch.allclose(calibration.symmetric_kl(a, a), torch.zeros(10), atol=1e-6)
    base, x, y = torch.randn(3, 10, 7, generator=gen)
    assert torch.allclose(calibration.composition_error(base, [x, y], x + y - base), torch.zeros(10), atol=1e-5)
    perfect = torch.full((100, 2), -20.0)
    perfect[:, 0] = 20.0
    assert calibration.argmax_ece(perfect, torch.zeros(100, dtype=torch.long)) < 1e-6
