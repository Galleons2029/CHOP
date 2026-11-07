import pytest
import torch

torch.set_num_threads(2)


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(1234)


@pytest.fixture(scope="session")
def tiny_model():
    from chop.tiny import tiny_qwen3

    return tiny_qwen3(attn_implementation="eager")
