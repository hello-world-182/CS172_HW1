"""Fast checks for the matched generator and the models (no test data is read)."""

import torch

from cs172.generate_captcha import generate_captcha
from cs172.models import MODELS, create_model


def main() -> None:
    for seed in range(50):
        image, label = generate_captcha(matched=True, seed=seed)
        assert image.size == (260, 80) and image.mode == "RGB" and len(label) == 5
    again, _ = generate_captcha(matched=True, seed=3)
    first, _ = generate_captcha(matched=True, seed=3)
    assert list(again.getdata()) == list(first.getdata()), "matched generator is not reproducible"
    print("generator: ok")

    images = torch.randint(0, 256, (4, 3, 80, 260), dtype=torch.uint8)
    targets = torch.randint(0, 10, (4, 5))
    for name in MODELS:
        model = create_model(name)
        logits = model(images)
        assert logits.shape == (4, 10, 5), (name, logits.shape)
        loss, _ = model.compute_loss(logits, targets)
        loss.backward()
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: output {tuple(logits.shape)}, loss {loss.item():.3f}, {params:,} parameters")


if __name__ == "__main__":
    main()
