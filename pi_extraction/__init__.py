"""Deterministic PI-source extractors (App. C.1).

p1 gold_patch    (oracle)      gold unified diff restricted to the files it modifies
p2 test_manifest (deployable)  FAIL_TO_PASS / PASS_TO_PASS test-function names
p3 next_file     (oracle)      next gold-modified file not yet edited + its top-level docstring
p4 call_graph    (deployable)  symbol-level call graph reachable from the issue-referenced module
math final_answer (oracle)     ground-truth final answer
"""
