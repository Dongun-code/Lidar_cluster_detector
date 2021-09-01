from typing import List, Sized
import torchvision.transforms as transforms
import numpy as np
import torch 
import matplotlib.pyplot as plt
from torch import Tensor
from image_list import ImageList
import cv2

class Transformation():
    def __init__(self, size):
        self.outsize = size
        self.size = (size,size)
        self.size_f = (size,size, 3)

    
    def __call__(self, image):
        tf = transforms.ToTensor()
        img_t = tf(image)
        # print("ttt:", img_t.shape)
        img = self.resize(img_t)
        # image_list = self.image_list(img)
        return img

    def image_list(self, image):
        # print('img shape:',image.shape)
        image_list = ImageList(image, image.shape)
        return image_list

    def pad_image(self, image):
        # pad_img = image.new(*self.size_f ).zero_()
        pad_img = torch.zeros((self.size_f))
        # print(pad_img)
        # print('-------------------------')
        pad_img[:image.shape[0], :image.shape[1],:image.shape[2]] = image
        # print('pad: ',pad_img)

        return pad_img

    def image_expansion(self, image: Tensor):
        #   return (3, 244, 244)
        # image = torch.nn.functional.interpolate(image[None], size=self.size, mode='bilinear', align_corners=False )[0]
        # print('imag: ', image)
        image = self.pad_image(image)
        image = image.permute(1,2,0)
        return image
        # print(image.shape)

        # plt.imshow(image)
        # plt.show()
        # pass    

    def image_reduce(self, image: Tensor):
        # image = cv2.resize(image, dsize=(244,244), interpolation=cv2.INTER_AREA)
        image = torch.nn.functional.interpolate(image[None], size=self.size, mode='bilinear', align_corners=False )[0]
        image = image.permute(1,2,0) 
        return image       
        # image.resize((244,244))
        # plt.imshow(image)
        # plt.show()       

    def resize(self, img):
        if np.min(img.shape[1:]) > self.outsize:
            #   use bilinear
            # print("244 >")
            img = self.image_reduce(img)
        elif np.min(img.shape[1:]) < self.outsize:
            img = self.image_expansion(img)
        # print('@@@@@@@@@@@@@@@: ', img.shape)
        return img

        