from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import functional as F

from typing import Callable

from datasets import load_flowers_dataset
from log import print_log
from models import BasicCNN


def train(
    epochs: int = 1000,
    # TODO use schedule
    lr: float = 5e-6,
    batch_size: int = 64,
    use_cuda: bool = True,
    log: Callable[[str], None] = print_log,
):
    train_loader = DataLoader(load_flowers_dataset(), batch_size, shuffle=True, num_workers=2, drop_last=True)
    log("Dataset loaded.")
    model = BasicCNN()
    optimiser = Adam(model.parameters(), lr)

    for epoch in range(epochs):
        for images in train_loader:
            if use_cuda:
                images = images.cuda()
            optimiser.zero_grad()

            output = model(images)

            loss = F.l1_loss(output, images)
            log(f"Reconstruction loss in epoch {epoch}: {loss}")

            loss.backward()
            optimiser.step()

        # TODO use test set
        # TODO visualise and log images


if __name__ == "__main__":
    train(use_cuda=False)
