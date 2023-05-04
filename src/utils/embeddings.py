from annoy import AnnoyIndex
import torch
from tqdm import tqdm


def create_img_embeddings(model, dataloader, device):
    """Returns a list of image embeddings."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for (img, label) in tqdm(dataloader, desc="Gallery embeddings"):
            img = img.to(device)
            img_feature = model(img)
            embeddings.append(img_feature.numpy().reshape(-1))

    return embeddings


def save_img_embeddings(embedding_list, save_path, embedding_dim, metric, n_trees):
    """Saves a list of image embeddings to an annoy index."""
    index = AnnoyIndex(embedding_dim, metric)
    for i, embedding in enumerate(embedding_list):
        index.add_item(i, embedding)

    index.build(n_trees=n_trees)
    index.save(save_path)
