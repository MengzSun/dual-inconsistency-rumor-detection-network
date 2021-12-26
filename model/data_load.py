from torch.utils import data
from PIL import Image
import os
import numpy as np
import torch

class Dataset_all(data.Dataset):
    def __init__(self,x_txt_input_ids,x_txt_attention_masks,x_img,x_kg1,x_kg2,x_kg_sim,y,transform,pathset):
        self.data_txt_input_ids = x_txt_input_ids
        self.data_txt_attention_masks = x_txt_attention_masks
        # self.data_txt_masks = x_txt_masks
        self.data_img = x_img
        self.data_kg1 = x_kg1
        self.data_kg2 = x_kg2
        self.data_kg_sim = x_kg_sim
        self.label = y
        self.transform = transform
        self.pathset = pathset
    def __getitem__(self,index):
        # img_path = '/home/sunmengzhu2019/kg_rumor/raw_data_pheme/pheme/images/'
        # if self.data_img[index] != 'none':
        img_path = self.pathset.path_img_data
        image = Image.open(
            img_path + self.data_img[index] + '.jpg'
        ).convert('RGB')
        image = self.transform(image)


        # else:
        #     image = Image.fromarray(128*np.ones((256,256,3),dtype = np.uint8))
        return self.data_txt_input_ids[index],self.data_txt_attention_masks[index], image,self.data_kg1[index],self.data_kg2[index],self.data_kg_sim[index],self.label[index]
    def __len__(self):
        return len(self.data_txt_input_ids)