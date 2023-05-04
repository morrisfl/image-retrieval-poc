from matplotlib import pyplot as plt
from PIL import Image
import torch
from torchvision import transforms


def get_indexes(model, img_path, embeddings, top_k, transform=None):
    """Return the indexes of the top_k most similar images to the query image."""
    img = Image.open(img_path)
    if transform:
        img = transform(img)
    else:
        img = transforms.ToTensor()(img)

    img = img.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        img_feature = model(img)
        img_feature = img_feature.numpy().reshape(-1)

    return embeddings.get_nns_by_vector(img_feature, top_k)


def visualize_retrieval_results(query_label, img_path, gallery_img, indexes):
    query_img = Image.open(img_path)
    size = query_img.size
    plt.figure(figsize=(int(size[0] / 3), int(size[1] / 3)))
    fig, ax = plt.subplots(nrows=3, ncols=3)

    # Query image
    ax[0, 0].imshow(query_img)
    ax[0, 0].set_title("Query image")
    ax[0, 0].axis("off")
    ax[0, 0].set_facecolor("white")
    ax[0, 1].axis("off")
    ax[0, 1].set_facecolor("white")
    ax[0, 2].axis("off")
    ax[0, 2].set_facecolor("white")

    # Top 6 retrieved images
    for i, index in enumerate(indexes):
        img = Image.open(gallery_img[index][0])
        img_size = img.size
        gt_label = gallery_img[index][1]
        ax[i // 3 + 1, i % 3].imshow(img)
        ax[i // 3 + 1, i % 3].set_title(f"Rank {i + 1}")
        ax[i // 3 + 1, i % 3].axis("off")
        ax[i // 3 + 1, i % 3].set_facecolor("white")
        if gt_label == query_label:
            rect = plt.Rectangle((0, 0), img_size[0], img_size[1], fill=False, edgecolor="green", linewidth=3)
            ax[i // 3 + 1, i % 3].add_patch(rect)
        else:
            rect = plt.Rectangle((0, 0), img_size[0], img_size[1], fill=False, edgecolor="red", linewidth=3)
            ax[i // 3 + 1, i % 3].add_patch(rect)
    plt.show()

