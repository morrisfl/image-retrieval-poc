from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import os


LANDMARKS = ["defense", "eiffel", "general", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame",
             "pantheon", "pompidou", "sacrecoeur", "triomphe"]


class Paris6kDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.labels = LANDMARKS
        self.query_data, self.query_img = self._get_query_data()
        self.gallery_data = self._get_gallery_data()

    def __len__(self):
        return len(self.gallery_data)

    def __getitem__(self, idx):
        img_path, label = self.gallery_data[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label

    def _get_query_data(self):
        """Return as dictionary with the label as key and a list of image paths as value."""
        data = {}
        query_img = []
        file_name = os.path.join(self.root, "query_images.txt")
        file = open(file_name, "r")
        for line in file.readlines():
            img_name = line.split()[0]
            label = line.split()[1]
            folder = img_name[6:]
            idx = folder.find("_")
            folder = folder[:idx]
            if label not in data:
                data[label] = []
            data[label].append(os.path.join(self.root, folder, img_name))
            query_img.append(img_name)

        return data, query_img

    def _get_gallery_data(self):
        """Return a list of tuples with the image path and the label."""
        data = []
        for label in self.labels:
            images = os.listdir(os.path.join(self.root, label))
            for img in images:
                if img not in self.query_img:
                    data.append((os.path.join(self.root, label, img), label))

        return data

