import os.path as op
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch._C import dtype
from torchvision import transforms


class Propose_region(torch.utils.data.Dataset):
    def __init__(self, img_set, labels, transform):
        self.img_set = img_set
        self.labels = labels
        self.transform = transform
        self.toTensor = transforms.ToTensor()

    def fill_padding(self, image):
        pad_img = np.zeros((224, 224, 3))

        # image_t = self.toTensor(image)
        # image_t = image_t.permute(1,2,0)
        image_n = np.array(image, dtype=np.float32) / 255
        # print('image_n:', image_n.shape)
        if image_n.shape[0] >= 224 and image_n.shape[1] <= 224:
            image_n = image_n[:224, :, :]
        elif image_n.shape[1] >= 224 and image_n.shape[0] <= 224:
            image_n = image_n[:, :224, :]
        elif image_n.shape[0] >= 224 and image_n.shape[1] >= 224:
            image_n = image_n[:224, :224, :]

        pad_img[:image_n.shape[0], :image_n.shape[1], :image_n.shape[2]] = image_n
        # print(pad_img)
        # plt.imshow(pad_img)
        # plt.show()
        pad_img = self.toTensor(pad_img).type(torch.FloatTensor)

        return pad_img

    def __getitem__(self, idx):
        img = self.img_set[idx]
        labels = self.labels[idx]
        # print('proposal :', img)
        img_shape = np.array(img).shape

        # if img_shape[0] >= 224 or img_shape[1] >=224:
        #     img = self.transform(img)
        # else:
        # print('labels:', labels)
        img = self.fill_padding(img)
        img = self.transform(img)

        return img, labels

    def __len__(self):
        return len(self.img_set)

# import os.path as op
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from torch._C import dtype
# from torchvision import transforms

# class Propose_region(torch.utils.data.Dataset):
#     def __init__(self, img_set, labels, transform):
#         self.img_set = img_set
#         self.labels = labels
#         self.transform = transform
#         self.toTensor = transforms.ToTensor()


#     def fill_padding(self, image):
#         pad_img = np.zeros((224,224,3))

#         # image_t = self.toTensor(image)
#         # image_t = image_t.permute(1,2,0)
#         image_n = np.array(image, dtype=np.float32) / 255

#         pad_img[:image_n.shape[0], :image_n.shape[1], :image_n.shape[2]] = image_n
#         # print(pad_img)
#         # plt.imshow(pad_img)
#         # plt.show()
#         pad_img = self.toTensor(pad_img).type(torch.FloatTensor)
#         # print(pad_img)
#         # pad_img = pad_img.permute((2, 0,1))
#         # print('pad', pad_img.shape)

#         return pad_img


#     def __getitem__(self, idx):
#         img = self.img_set[idx]
#         # print('proposal :', img)

#         img_shape = np.array(img).shape

#         if img_shape[0] >= 224 or img_shape[1] >=224:
#             img = self.transform(img)
#         else:
#             img = self.fill_padding(img)

#         labels = self.labels[idx]


#         # print('out img shape:', img.shape)
#         return img, labels

#     def __len__(self):
#         return len(self.img_set)
