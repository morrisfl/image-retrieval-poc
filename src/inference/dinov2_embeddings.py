from annoy import AnnoyIndex
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data import Paris6kDataset
from src.models import DINOv2Model
from src.utils.embeddings import create_img_embeddings, save_img_embeddings

# Configs ---------------------------------------------------------------------------------------
DATA_PATH = "../../dataset/paris6k"
MODEL_NAME = "dinov2_vits14"
EMBEDDING_SIZE = 384
INDEX_METRIC = "angular"
INDEX_TREE = 25
INDEX_SAVE_PATH = f"../../embeddings/index{INDEX_TREE}_dinov2_vits14_paris6k.ann"
# -----------------------------------------------------------------------------------------------

# Image transforms [Source: https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py]
transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Dataset
dataset = Paris6kDataset(root=DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load pretrained DINOv2 model
model = DINOv2Model(model_name=MODEL_NAME)
model.to(device)

# Create image embeddings
embeddings = create_img_embeddings(model=model, dataloader=dataloader, device=device)

# Create annoy index
save_img_embeddings(embedding_list=embeddings, save_path=INDEX_SAVE_PATH, embedding_dim=EMBEDDING_SIZE,
                    metric=INDEX_METRIC, n_trees=INDEX_TREE)
