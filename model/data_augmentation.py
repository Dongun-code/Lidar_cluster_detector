import torch
from PIL import Image, ImageEnhance, ImageChops
import matplotlib.pyplot as plt
import random

def transform_img(img, label):
    augment_img = []
    augment_label = []

    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_vertical = img.transpose(Image.FLIP_TOP_BOTTOM)
    rotate_image = img.rotate(random.randint(-20, 20))

    width, height = img.size
    shift = random.randint(0, int(height * 0.1))
    vertical_shift_image = ImageChops.offset(img, 0, shift)
    vertical_shift_image.paste((0), (0, 0, width, shift))

    shift = random.randint(0, int(width * 0.2))
    horizonal_shift_image = ImageChops.offset(img, shift, 0)
    horizonal_shift_image.paste((0), (0, 0, shift, height))
    augment_img.append([img_flip, img_vertical, rotate_image, vertical_shift_image, horizonal_shift_image])
    augment_label.append([label, label, label, label, label])
    augment_img = augment_img[0]
    augment_label = augment_label[0]

    return augment_img, augment_label


def data_augmentation(images, labels, device):
    #   category index
    augmentation_category = [2, 3]
    labels = labels.to('cpu').tolist()
    for index in range(len(images)):
        img = images[index]
        label = labels[index]
        print("labels : ", label)
        # plt.imshow(img)
        # plt.show()
        if label in augmentation_category:
            augment_img, augment_label = transform_img(img, label)
            images += augment_img
            labels += augment_label
            print(images)

    labels = torch.tensor(labels).to(device)

    return images, labels



