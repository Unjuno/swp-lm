"""Synthetic invariant tests; no user-provided model weights required."""
import torch
from swp_lm import StudentLM, decode_pq, make_freq_bias


def fixture(vocab=41):
    torch.manual_seed(42)
    centers = torch.randn(2, 8, 128, 8, dtype=torch.float32) * 0.02
    codes = torch.randint(0, 128, (vocab, 2, 8), dtype=torch.long)
    table = decode_pq(centers, codes)
    inputs = torch.randn(vocab, 64) * 0.05
    bias = torch.randn(vocab) * 0.001
    return inputs, centers, codes, table, bias


def test_pq_decoder_shape_and_equality():
    _, centers, codes, table, _ = fixture()
    assert table.shape == (41, 64)
    for i in [0, 1, 7, 40]:
        expected = torch.cat([centers[0, group, codes[i, 0, group]]
                             + centers[1, group, codes[i, 1, group]] for group in range(8)])
        assert torch.allclose(table[i], expected, atol=0, rtol=0)


def test_pq_student_forward_and_normalization():
    args = fixture()
    ids = torch.randint(0, 41, (2, 5))
    model = StudentLM('pq_frozen', *args)
    logits = model(ids)
    assert logits.shape == (2, 5, 41)
    assert torch.isfinite(logits).all()
    probs = logits.softmax(-1)
    assert torch.allclose(probs.sum(-1), torch.ones(2, 5), atol=1e-6)


def test_trainable_dictionary_gradients():
    args = fixture()
    model = StudentLM('pq_learn', *args)
    ids = torch.randint(0, 41, (2, 5))
    loss = torch.nn.functional.cross_entropy(model(ids).flatten(0, 1), ids.flatten())
    loss.backward()
    assert model.centers.grad is not None
    assert torch.isfinite(model.centers.grad).all()


def test_frequency_bias_is_finite():
    bias = make_freq_bias(torch.tensor([0, 0, 9, 3]))
    assert torch.isfinite(bias).all()
    assert torch.allclose(bias.exp().sum(), torch.tensor(1.), atol=1e-6)
