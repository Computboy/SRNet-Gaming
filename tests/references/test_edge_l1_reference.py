"""Run `python test_edge_l1_reference.py` from this folder."""
import torch
import torch.nn.functional as F
from edge_l1_reference import hr_edge_mask, edge_region_l1, combined_l1_loss


def test_mask_and_loss():
    # 1. No edges -> zero edge term and equality to global L1.
    flat = torch.full((2, 3, 16, 16), 0.5)
    sr = torch.full_like(flat, 0.6, requires_grad=True)
    m = hr_edge_mask(flat)
    assert m.sum().item() == 0
    total, global_l1, edge_l1, coverage = combined_l1_loss(sr, flat, 0.1)
    assert edge_l1.item() == 0
    assert coverage.item() == 0
    assert torch.allclose(total, global_l1)

    # 2. Step in HR -> positive edge band; no outside border.
    hr = torch.zeros(2, 3, 16, 16)
    hr[:, :, :, 8:] = 1.0
    mask = hr_edge_mask(hr)
    assert mask.sum() > 0
    assert mask[:, :, 0, :].sum() == 0
    assert mask[:, :, -1, :].sum() == 0
    assert mask[:, :, :, 0].sum() == 0
    assert mask[:, :, :, -1].sum() == 0
    assert torch.equal(mask, hr_edge_mask(hr.clone()))

    # 3. A masked error produces an edge loss and gradient.
    pred = hr.detach().clone()
    pred[:, :, :, 7:9] = 0.5
    pred.requires_grad_(True)
    total, global_l1, edge_l1, cov = combined_l1_loss(pred, hr, 0.1)
    assert edge_l1.item() > 0
    assert cov.item() > 0
    total.backward()
    assert pred.grad is not None and pred.grad.abs().sum().item() > 0

    # 4. Lambda=0 yields original global L1 to numerical precision.
    total0, glob0, edge0, _ = combined_l1_loss(pred.detach(), hr, 0)
    assert torch.allclose(total0, F.l1_loss(pred.detach(), hr))
    assert torch.allclose(glob0, total0)
    print("PASS: empty masks, border, deterministic mask, gradient, lambda=0 parity")


if __name__ == "__main__":
    test_mask_and_loss()
