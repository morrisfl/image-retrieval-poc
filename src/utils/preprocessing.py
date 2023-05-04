import os
from PIL import Image


def remove_invalid_images():
    """Remove invalid images from the dataset"""
    for root, dirs, files in os.walk('../../dataset'):
        for file in files:
            if file.endswith('.jpg'):
                try:
                    img = Image.open(os.path.join(root, file))
                    img.verify()
                except (IOError, SyntaxError) as e:
                    print('Bad file:', os.path.join(root, file))
                    os.remove(os.path.join(root, file))


if __name__ == '__main__':
    remove_invalid_images()
