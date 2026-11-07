import numpy as np
import torch

from chop.quantile import StreamingQuantile


def test_per_state_thresholds_match_numpy(gen):
    sq = StreamingQuantile(window=50, refresh=1, min_history=10)
    hist = torch.rand(30, generator=gen)
    sq.update(hist)
    new = torch.rand(40, generator=gen)
    th = sq.thresholds_for_stream(new, 0.3).numpy()
    stream = torch.cat([hist, new]).numpy()
    for i in range(40):
        end = 30 + i
        ref = np.quantile(stream[max(0, end - 50):end], 0.3)
        assert abs(th[i] - ref) < 1e-6


def test_block_refresh_and_warmup(gen):
    sq = StreamingQuantile(window=100, refresh=8, min_history=5)
    new = torch.rand(20, generator=gen)
    th = sq.thresholds_for_stream(new, 0.5)
    assert torch.isnan(th[:8]).all()                   # nothing before the first block
    ref = torch.quantile(new[:8], 0.5)
    assert torch.allclose(th[8:16], ref.expand(8))     # constant within a refresh block
    assert len(sq) == 0                                # thresholds_for_stream does not mutate


def test_update_keeps_last_window(gen):
    sq = StreamingQuantile(window=10, refresh=4, min_history=1)
    sq.update(torch.arange(25.0))
    assert torch.equal(sq.history, torch.arange(15.0, 25.0))
    state = sq.state_dict()
    other = StreamingQuantile()
    other.load_state_dict(state)
    assert torch.equal(other.history, sq.history) and other.window == 10
