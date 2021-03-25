from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn import functional as F

from typing import Callable, Optional

from datasets import load_flowers_dataset
from log import initialise_logging
from models import BasicCNN


def train(
    log_dir: str,
    epochs: int = 1000,
    initial_lr: float = 1e-4,
    lr_scheduler_patience: int = 10,
    batch_size: int = 64,
    use_cuda: bool = True,
    log_images_after_epochs: Optional[int] = 10
):
    log, log_images = initialise_logging(log_dir)
    train_loader = DataLoader(load_flowers_dataset(), batch_size, shuffle=True, num_workers=2, drop_last=True)
    log("Dataset loaded.")

    model = BasicCNN()
    if use_cuda:
        model.cuda()

    optimiser = Adam(model.parameters(), initial_lr)
    lr_scheduler = ReduceLROnPlateau(optimiser, patience=lr_scheduler_patience, verbose=True)

    for epoch in range(1, epochs + 1):
        loss_sum = 0

        for images in train_loader:
            if use_cuda:
                images = images.cuda()
            optimiser.zero_grad()

            output = model(images)

            loss = F.l1_loss(output, images)
            loss.backward()
            optimiser.step()
            loss_sum += loss.detach().item()

        loss_mean = loss_sum / len(train_loader)
        log(f"[Epoch {epoch}] Train loss: {loss_mean}")

        # TODO validate

        lr_scheduler.step(loss_mean)  # TODO give validation loss as input instead

        if log_images_after_epochs is not None and epoch % log_images_after_epochs == 0:
            # TODO log images
            pass


if __name__ == "__main__":
    train("../log/")
