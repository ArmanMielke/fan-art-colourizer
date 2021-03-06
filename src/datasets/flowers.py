from .transform import get_transform
from .image_dataset import ImageDataset


def load_flowers_dataset(path: str = "../data/102flowers/jpg", image_size: int = 64) -> ImageDataset:
    """
    Loads the 102 Category Flower Dataset (https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html),
    without any labels.
    """
    return ImageDataset(root=path, transform=get_transform(image_size))
