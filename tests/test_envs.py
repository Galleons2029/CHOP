from envs.math.reward import extract_answer, is_correct, last_boxed
from envs.toy import ToyArithmetic, ToyTokenizer


def test_answer_extraction():
    assert last_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert extract_answer("work\nAnswer: 42") == "42"
    assert extract_answer(r"Answer: 3 then \boxed{7}") == "7"
    assert extract_answer("no answer here") is None


def test_is_correct():
    assert is_correct("...\nAnswer: 204", "204")
    assert is_correct(r"\boxed{204.0}", "204")
    assert not is_correct("Answer: 205", "204")


def test_toy_task_roundtrip():
    tok = ToyTokenizer()
    assert tok.decode(tok.encode("Q:1+2=? A:")) == "Q:1+2=? A:"
    task = ToyArithmetic(n_train=4, n_eval=2)
    inst = task.train_instances()[0]
    assert task.reward(inst, inst.answer + " tail") == 1.0
    assert set(task.pi_blocks(inst)) == {"final_answer", "hint"}


def test_prepare_math_rows_carry_teacher_and_pi_ids():
    import importlib.util

    spec = importlib.util.spec_from_file_location("prep", "scripts/prepare_math.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    from envs.toy import ToyArithmetic, ToyTokenizer

    tok, task = ToyTokenizer(), ToyArithmetic(n_train=3)
    rows = prep.build_rows(task.train_instances(), task, tok.encode, lambda t: tok.encode("T" + t), tok.encode,
                           ["final_answer"], "toy")
    r = rows[0]["extra_info"]
    assert r["teacher_prompt_ids"][0] == tok.encode("T")[0]
    assert set(r["pi_prefix_ids"]) == {"final_answer"}
    assert r["plan_prompt_len"] == len(r["teacher_prompt_ids"]) + len(r["pi_prefix_ids"]["final_answer"])
