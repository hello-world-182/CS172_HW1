"""Image datasets for directory and ZIP based CAPTCHA collections."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class CAPTCHA(Dataset):
    """Load RGB images and five-digit labels from a directory or ZIP file."""

    def __init__(self, source, target_size=(80, 260), transform=None, cache_images=False, num_images=None):
        self.source = os.fspath(source)
        self.target_size = tuple(target_size) if target_size is not None else None
        self.transform = transform
        self.cache_images = cache_images
        path = Path(self.source)
        self._is_zip = path.is_file() and path.suffix.lower() == ".zip"
        self._archive = None
        if self._is_zip:
            with zipfile.ZipFile(path, "r") as archive:
                self.img_paths = sorted(
                    name for name in archive.namelist()
                    if Path(name).suffix.lower() in IMAGE_SUFFIXES and not name.endswith("/")
                )
        else:
            if not path.is_dir():
                raise FileNotFoundError(f"image directory or ZIP does not exist: {self.source}")
            self.img_paths = sorted(
                str(item) for item in path.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            )
        if num_images is not None:
            if num_images < 1:
                raise ValueError("num_images must be positive")
            self.img_paths = self.img_paths[:num_images]
        self._cache = {}

    @staticmethod
    def _parse_label(filename):
        digits = Path(filename).stem.split("#", maxsplit=1)[0]
        if len(digits) != 5 or not digits.isdigit():
            raise ValueError(f"expected a five-digit label in filename {filename!r}")
        return torch.tensor([int(value) for value in digits], dtype=torch.long)

    def _read_image(self, image_path):
        if image_path in self._cache:
            return self._cache[image_path].clone()
        if self._is_zip:
            if self._archive is None:
                self._archive = zipfile.ZipFile(self.source, "r")
            source = self._archive.open(image_path)
        else:
            source = image_path
        with Image.open(source) as image:
            image = image.convert("RGB")
            if self.target_size is not None:
                height, width = self.target_size
                image = image.resize((width, height), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if self.cache_images:
            self._cache[image_path] = tensor
        return tensor

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archive"] = None
        return state

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]
        image = self._read_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, self._parse_label(path)


class ZIP_CAPTCHA(CAPTCHA):
    """Explicit name for the ZIP-backed CAPTCHA dataset."""

