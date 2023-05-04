from annoy import AnnoyIndex
import torch
from torch.utils.data import DataLoader

from src.data import Paris6kDataset
from src.models import CLIPModelPretrained
from src.utils.embeddings import create_img_embeddings, save_img_embeddings

# Configs ---------------------------------------------------------------------------------------
DATA_PATH = "../../dataset/paris6k"
MODEL_NAME = "ViT-B/32"
PRETRAINED_MODEL = "laion2b_s34b_b79k"
INDEX_METRIC = "angular"
INDEX_TREE = 25
INDEX_SAVE_PATH = f"../../embeddings/index{INDEX_TREE}_clip_ViT-B-32_laion2B_paris6k.ann"
# -----------------------------------------------------------------------------------------------

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load pretrained CLIP model
model = CLIPModelPretrained(model_name=MODEL_NAME, pretrained_model=PRETRAINED_MODEL)
model.to(device)

# Image transforms
transform = model.preprocess

# Dataset
dataset = Paris6kDataset(root=DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Create image embeddings
embeddings = create_img_embeddings(model=model, dataloader=dataloader, device=device)

# Create and save annoy index
save_img_embeddings(embedding_list=embeddings, save_path=INDEX_SAVE_PATH, embedding_dim=512, metric=INDEX_METRIC,
                    n_trees=INDEX_TREE)
