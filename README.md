# Image Retrieval (Proof of Concept)
## Introduction
This is a proof of concept for a image retrieval system. The system is based on the 
[Paris 6k Dataset](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) and the image embeddings can be generated 
using three different pretrained model: [open_clip](https://github.com/mlfoundations/open_clip), 
[DINO](https://github.com/facebookresearch/dino) and [DINOv2](https://github.com/facebookresearch/dinov2). The generated
embeddings are stored in an [Annoy Index](https://github.com/spotify/annoy) in the embeddings folder and can be used to
retrieve similar images to a given query image (examples can be seen under [src/notebooks](src/notebooks)).

## Installation
The project can be installed within a virtual environment using the following command. The example uses conda and the 
pytorch installation is based on the [pytorch website](https://pytorch.org/get-started/locally/) for MacOS. 
```
conda create -n image_retrieval python=3.8
conda activate image_retrieval
conda istall conda install pytorch torchvision -c pytorch
pip install .
```

## Dataset
The dataset can be downloaded from [here](https://www.kaggle.com/datasets/skylord/oxbuildings) and should
be extracted to the [dataset/paris6k](dataset/paris6k) folder. The folder structure should look like this:
```
dataset
├── paris6k
│   ├── defense
│   ├── eiffel
│   ├── general
│   ├── invalides
│   ├── louvre
│   ├── moulinrouge
│   ├── museedorsay
│   ├── notre_dame
│   ├── pantheon
│   ├── pompidou
│   ├── sacrecoeur
│   ├── triomphe
│   └── query_images.txt
```
Since the dataset contains erroneous images, please run the following command to remove them:
```
python src/utils/preprocessing.py
```
