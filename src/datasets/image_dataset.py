from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

import os
from typing import Callable, List


class ImageDataset(Dataset):
    images: List[Tensor]

    def __init__(self, root: str, transform: Callable):
        with os.scandir(root) as dir_entries:
            image_paths = [entry.path for entry in dir_entries if entry.is_file()]

        # TODO decide whether the images should be loaded at the start or on demand
        self.images = [transform(Image.open(path)) for path in image_paths]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
