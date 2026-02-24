from sklearn.model_selection import train_test_split
import torch
import nibabel as nib
import numpy as np
import os
import json
import pandas as pd
from torchvision import datasets, transforms
from sklearn.preprocessing import OneHotEncoder
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    CenterSpatialCrop,
    ScaleIntensityRange,
    AddChanneld,
    RandRotate,
    Orientation,
    ToTensor,
    RandAffine
)
from torch.utils.data import DataLoader, Dataset
from scipy.stats import norm
import random
import SimpleITK as sitk


def load_image(path):
    
    img = sitk.GetArrayFromImage(sitk.ReadImage(path))
    img_out = np.expand_dims(img, axis = -1)
    
    return img_out

def crop_center(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example: 
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop

def get_heathy_outcome_label(whole_label_path,outcome_path):

    with open(whole_label_path,'r', encoding='UTF-8') as f:
        label_dict = json.load(whole_label_path)
        
    outcome_df = pd.read_csv(outcome_path)
    outcome_df = outcome_df[outcome_df["f.eid"].astype(str).isin(list(label_dict.keys()))]
    healthy_df = outcome_df[outcome_df.apply(lambda row: "F0" not in str(row), axis=1)]
    healthy_fids = [str(i) for i in list(healthy_df["f.eid"])]
    healthy_label_dict = {}
    disease_label_dict = {}
    for k,v in label_dict.items():
        if k in healthy_fids:
            healthy_label_dict[k] = v
        else:
            disease_label_dict[k] = v
    return healthy_label_dict,disease_label_dict


def num2vect(x, bin_range, bin_step, sigma):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)
    

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        
def create_dataset(args):
    path = args["data_path"]
    aug = args["augmentation"]
    batch_size = args["batch_size"]
    used_dist = args["distribution"]
    corr = args["corr"]
    
    
    # with open(path,'r', encoding='UTF-8') as f:
    #     data_dict = json.load(f)
    
    brain_t1_mri_path = path
    df = pd.read_csv(brain_t1_mri_path)
    data_dict = {}
    for i in list(df["eid"]):
        img = df[df["eid"] == i]["T1_PATH"].item()
        label = df[df["eid"] == i]["Age"].item()
        condition = df[df["eid"] == i]["Gender"].item()
        data_dict[img] = [label,condition]

    pin_memory = True
    images = list(data_dict.keys())
    
    train_images, val_test_images = train_test_split(images,test_size=0.3, random_state=42)
    val_images,test_images = train_test_split(val_test_images,test_size=0.8, random_state=42)

    train_dict = {img: data_dict[img] for img in train_images}
    val_dict = {img: data_dict[img] for img in val_images}
    test_dict = {img: data_dict[img] for img in test_images}
    
    ds_train = MriDataset(train_dict,aug,used_dist,corr)
    # close aug to eval and test
    aug = False
    ds_val = MriDataset(val_dict, aug,used_dist,corr)
    ds_test = MriDataset(test_dict, aug,used_dist,corr)
    
    dl_train = DataLoader(ds_train, batch_size = batch_size,num_workers=28, shuffle=True,pin_memory=pin_memory,drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size,num_workers=28, pin_memory=pin_memory,drop_last=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size,num_workers=28, pin_memory=pin_memory,drop_last=True)
    
    return ds_train,ds_val,ds_test,dl_train,dl_val,dl_test


def create_pre_splited_dataset(args):
    path = args["data_path"]
    aug = args["augmentation"]
    batch_size = args["batch_size"]
    used_dist = args["distribution"]
    corr = args["corr"]
    aug = False
    pin_memory = True
    
    with open('./ukb_dataset/train.json', 'r', encoding='utf-8') as f:
        train_dict = json.load(f)
        
    with open('./ukb_dataset/val.json', 'r', encoding='utf-8') as f:
        val_dict = json.load(f)
        
    with open('./ukb_dataset/test.json', 'r', encoding='utf-8') as f:
        test_dict = json.load(f)
    
    ds_train = MriDataset(train_dict, aug,used_dist,corr)
    ds_val = MriDataset(val_dict, aug,used_dist,corr)
    ds_test = MriDataset(test_dict, aug,used_dist,corr)
    
    dl_train = DataLoader(ds_train, batch_size = batch_size,num_workers=28, shuffle=True,pin_memory=pin_memory,drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size,num_workers=28, pin_memory=pin_memory,drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size,num_workers=28, pin_memory=pin_memory,drop_last=False)
    
    return ds_train,ds_val,ds_test,dl_train,dl_val,dl_test


def create_external_dataset(args):
    path = args["data_path"]
    aug = args["augmentation"]
    batch_size = args["batch_size"]
    used_dist = args["distribution"]
    corr = args["corr"]
    
    
    pin_memory = True
    with open("./external_dataset_adni/train.json",'r', encoding='UTF-8') as f:
        train_dict = json.load(f)
        
    with open("./external_dataset_adni/val.json",'r', encoding='UTF-8') as f:
        val_dict = json.load(f)
        
    with open("./external_dataset_adni/test.json",'r', encoding='UTF-8') as f:
        test_dict = json.load(f)
    
    # close aug to eval and test
    aug = False
    ds_train = MriDataset(train_dict,aug,used_dist,corr)
    ds_val = MriDataset(val_dict, aug,used_dist,corr)
    ds_test = MriDataset(test_dict, aug,used_dist,corr)
    
    dl_train = DataLoader(ds_train, batch_size = batch_size,num_workers=28, shuffle=True,pin_memory=pin_memory,drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size,num_workers=28, pin_memory=pin_memory,drop_last=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size,num_workers=28, pin_memory=pin_memory,drop_last=True)
    
    return ds_train,ds_val,ds_test,dl_train,dl_val,dl_test


class MriDataset(Dataset):
    def __init__(self, data_dict,aug,dist,corr):
       
        self.data_dict = data_dict
        self.aug = aug
        self.dist = dist
        self.corr = corr
        self.img_ls = list(self.data_dict.keys())
        self.resize_transform = Resize(spatial_size=(96, 96, 96), mode='trilinear')
        self.center_crop = CenterSpatialCrop(roi_size=(160, 192, 160))
        self.rand_rot =  RandAffine(
            prob=1.0,
            translate_range=(40, 40, 2),
            rotate_range=(np.pi / 36, np.pi / 36, np.pi / 4),
            scale_range=(0.15, 0.15, 0.15),
            padding_mode="border",
        )

        self.ensureChannelFirst = EnsureChannelFirst(channel_dim=-1)
        self.ori = Orientation(axcodes="RAS")
        self.scale_intensity_transform = ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.img_ls)

    def __getitem__(self, idx):
        img_path = self.img_ls[idx]
        image = load_image(img_path)
        age = self.data_dict[img_path][0]
        gender = self.data_dict[img_path][1]

        if self.aug:
            image = self.ensureChannelFirst(image)
            image = self.rand_rot(image)
            image = self.center_crop(image)
            image = self.ori(image)
            image = self.scale_intensity_transform(image)
            image = self.to_tensor(image)
        
        else:
            image = self.ensureChannelFirst(image)
            image = self.center_crop(image)
            # image = self.resize_transform(image)
            image = self.ori(image)
            image = self.scale_intensity_transform(image)
            image = self.to_tensor(image)
            
        if self.dist:
            bin_range = [44,84]
            bin_step = 1
            sigma = 1
            y, bc = num2vect(age, bin_range, bin_step, sigma)
            if self.corr:
                gender = torch.tensor(gender)
                return image,y,bc,age,gender
            
            else:
                return image,y,bc,age
        
        else:  
            return image,age