import torch

from adaptkpn import AdaptKPN, frozen_args, train_adaptkpn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(1, 1, 32, 32, device=device)
    model = AdaptKPN(n_chan=1).to(device)
    y = model(x)
    assert y.shape == x.shape
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 2649

    cfg = frozen_args(epochs=1, amp=False, print_every=1)
    trained, loss = train_adaptkpn(x, cfg, device)
    with torch.no_grad():
        z = trained(x)
    assert z.shape == x.shape
    assert torch.isfinite(torch.tensor(loss))
    print("smoke test passed")


if __name__ == "__main__":
    main()
