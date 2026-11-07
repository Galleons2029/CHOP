"""verl (FSDP2 + vLLM) recipe for CHOP.

``chop_verl.compute`` holds the per-rank computations and is importable without Ray or verl;
``chop_verl.worker`` / ``chop_verl.trainer`` / ``chop_verl.main`` wire them into verl 0.9.x.
"""
