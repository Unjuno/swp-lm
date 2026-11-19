"""Self-contained demo without pretrained assets."""
import torch
from swp_lm import StudentLM, decode_pq


def main():
    torch.manual_seed(0)
    vocab = 100
    centers = torch.randn(2, 8, 128, 8) * 0.02
    codes = torch.randint(128, (vocab, 2, 8))
    output_table = decode_pq(centers, codes)
    inputs = torch.randn(vocab, 64) * 0.02
    model = StudentLM('pq_frozen', inputs, centers, codes, output_table, torch.zeros(vocab))
    logits = model(torch.randint(vocab, (1, 8)))
    print('logits shape:', tuple(logits.shape))
    print('max normalization error:', (logits.softmax(-1).sum(-1)-1).abs().max().item())


if __name__ == '__main__':
    main()
