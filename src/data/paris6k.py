import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import os
import random


GT_PATH = "gt_labels"
IMG_QUALITY = ["good", "ok", "junk"]
LANDMARKS = ["defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame",
             "pantheon", "pompidou", "sacrecoeur", "triomphe"]


class Paris6kDataset:
    def __init__(self, root, query):
        self.root = root
        self.labels = LANDMARKS
        self.data = self.get_dataset(query=query)

    def get_dataset(self, query):
        if query:
            return self._get_query_img()
        else:
            data = self._get_gallery_img()
            dataset = []
            for label in data:
                for quality in data[label]:
                    for img in data[label][quality]:
                        dataset.append((img, label))
            return dataset

    def _get_query_img(self):
        """Return as dictionary with the label as key and a list of image paths as value."""
        data = {}
        for label in self.labels:
            data[label] = []
            for i in range(1, 6):
                path = os.path.join(self.root, GT_PATH, f"{label}_{i}_query.txt")
                file = open(path, "r")
                img_name = file.read().split()[0] + ".jpg"
                name = img_name[6:]
                idx = name.find("_")
                name = name[:idx]
                data[label].append(os.path.join(self.root, name, img_name))

        return data

    def _get_gallery_img(self):
        """Return a nested dictionary with the label as key and a dictionary with the image quality as key and a list
        of image paths as value."""
        data = {}
        for label in self.labels:
            data[label] = {}
            for quality in IMG_QUALITY:
                data[label][quality] = []
                path = os.path.join(self.root, GT_PATH, f"{label}_1_{quality}.txt")
                file = open(path, "r")
                for line in file.readlines():
                    img_name = line.split()[0] + ".jpg"
                    name = line[6:]
                    idx = name.find("_")
                    name = name[:idx]
                    data[label][quality].append(os.path.join(self.root, name, img_name))

        return data
