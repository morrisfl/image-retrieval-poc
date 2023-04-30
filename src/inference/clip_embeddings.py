from annoy import AnnoyIndex
from PIL import Image
import torch
from tqdm import tqdm

from src.data import Paris6kDataset
from src.models import CLIPModelPretrained

# Configs ---------------------------------------------------------------------------------------
DATA_PATH = "../../dataset/paris6k"
MODEL_NAME = "ViT-B/32"
PRETRAINED_MODEL = "laion2b_s34b_b79k"
INDEX_METRIC = "angular"
INDEX_TREE = 25
INDEX_SAVE_PATH = f"../../embeddings/index{INDEX_TREE}_clip_ViT-B-32_laion2B_paris6k.ann"
# -----------------------------------------------------------------------------------------------

# Dataset
gallery_dataset = Paris6kDataset(root=DATA_PATH, query=False)
gallery_data = gallery_dataset.data

# Load pretrained CLIP model
model = CLIPModelPretrained(model_name=MODEL_NAME, pretrained_model=PRETRAINED_MODEL)

# Create image embeddings
model.eval()
embeddings = []
with torch.no_grad():
    for (img, label) in tqdm(gallery_data, desc="Gallery embeddings"):
        img = Image.open(img)
        img_feature = model(img)
        embeddings.append(img_feature.numpy().reshape(-1))

# Create annoy index
index = AnnoyIndex(512, INDEX_METRIC)
for i, embedding in enumerate(embeddings):
    index.add_item(i, embedding)

index.build(n_trees=INDEX_TREE)
index.save(INDEX_SAVE_PATH)






