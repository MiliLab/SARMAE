# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import random
import numpy as np
import PIL
import torch
from torch import NoneType

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


from PIL import Image,ImageFile
from torch.utils import data
from torch.utils.data import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class SAROpticalUnpairedDataset(Dataset):

    def __init__(self,
                 sar_dir,
                 optical_dir,
                 input_size=224,
                 max_samples=None,
                 noise_std=0.1):

        super().__init__()
        self.sar_paths = self._gather_images(sar_dir, max_samples)
        self.optical_paths = self._gather_images(optical_dir, max_samples)
        self.noise_std = noise_std

        self.sar_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # [3, H, W]
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.optical_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print(f"[SAROpticalUnpairedDataset] Loaded {len(self.sar_paths)} SAR images and {len(self.optical_paths)} Optical images.")

    def _gather_images(self, folder, max_samples=None):
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        files = [os.path.join(folder, f)
                 for f in os.listdir(folder)
                 if f.lower().endswith(valid_exts)]
        random.shuffle(files)
        if max_samples:
            files = files[:max_samples]
        return files

    def __len__(self):
        return max(len(self.sar_paths), len(self.optical_paths))

    def __getitem__(self, idx):

        sar_path = random.choice(self.sar_paths)
        optical_path = random.choice(self.optical_paths)

        sar_img = Image.open(sar_path).convert('L')
        sar_clean = sar_img.copy()
        sar_noisy = self.add_noise(sar_img)

        sar_clean = self.sar_transform(sar_clean)
        sar_noisy = self.sar_transform(sar_noisy)

        optical_img = Image.open(optical_path).convert('RGB')
        optical_img = self.optical_transform(optical_img)

        return sar_noisy, sar_clean, optical_img

    def add_noise(self, img):

        arr = np.array(img).astype(np.float32) / 255.0 + 1e-6  # 避免 log(0)
        log_arr = np.log(arr)

        noise = np.random.normal(0, self.noise_std, log_arr.shape)
        log_noisy = log_arr + noise

        noisy = np.exp(log_noisy)
        noisy = np.clip(noisy, 0, 1)

        noisy_img = Image.fromarray((noisy * 255).astype(np.uint8))
        return noisy_img
    
    
class MillionAIDDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=500_000):
        self.root_dir = root_dir
        self.transform = transform

        self.img_files = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'))
        ]
        np.random.shuffle(self.img_files)
        if len(self.img_files) > max_samples:
            self.img_files = self.img_files[:max_samples]
        print(f"MillionAIDDataset: loaded {len(self.img_files)} images from {root_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert('RGB')  # 保证三通道
        if self.transform:
            img = self.transform(img)
        return img

class RAWSARDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform

        self.img_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp')):
                    self.img_paths.append(os.path.join(subdir, file))
        np.random.shuffle(self.img_paths)
        if max_samples is not None and len(self.img_paths) > max_samples:
            self.img_paths = self.img_paths[:max_samples]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img

class SAR30Dataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform

        self.img_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp')):
                    self.img_paths.append(os.path.join(subdir, file))
        np.random.shuffle(self.img_paths)
        if max_samples is not None and len(self.img_paths) > max_samples:
            self.img_paths = self.img_paths[:max_samples]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img
    
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset == 'millionaid':
        print('Loading MillionAID dataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/optical_all_img/'
        args.nb_classes = 51
        dataset = MillionAIDDataset(data_path, train=is_train, transform=transform, tag=args.tag)
    
    elif args.dataset == 'sar_acd_30':
        print('Loading SAR_ACD dataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/use/cls/SAR_ACD/SAR_ACD_30/Train'
        args.nb_classes = 5
        root = os.path.join(args.data_path, 'Train' if is_train else 'Val')
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        return dataset
    
    elif args.dataset == 'sar_acd_40':
        print('Loading SAR_ACD_40 dataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/use/cls/SAR_ACD/SAR_ACD_40/Train'
        args.nb_classes = 5
        root = os.path.join(args.data_path, 'Train' if is_train else 'Val')
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        return dataset
    
    elif args.dataset == 'fusar':
        print('Loading fusar dataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/use/cls/New_FUSAR/Train'
        args.nb_classes = 10
        root = os.path.join(args.data_path, 'Train' if is_train else 'Val')
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        return dataset
    
    elif args.dataset == 'fusar_10':
        print('Loading fusar_10 dataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/use/cls/New_FUSAR/FUSAR_10/Train'
        args.nb_classes = 10
        root = os.path.join(args.data_path, 'Train' if is_train else 'Val')
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        return dataset
    
    elif args.dataset == 'fusar_20':
        print('Loading fusar_20 dataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/use/cls/New_FUSAR/FUSAR_20/Train'
        args.nb_classes = 10
        root = os.path.join(args.data_path, 'Train' if is_train else 'Val')
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        return dataset
    
    elif args.dataset == 'fusar_40':
        print('Loading fusar_40 dataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/use/cls/New_FUSAR/FUSAR_40/Train'
        args.nb_classes = 10
        root = os.path.join(args.data_path, 'Train' if is_train else 'Val')
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        return dataset
    
    elif args.dataset == 'fusar_40_dinov3':
        print('Loading fusar_40 dataset for DINOv3!')
        args.nb_classes = 10
        root = os.path.join(args.data_path, 'Train' if is_train else 'Val')
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if is_train:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),  
                transforms.Resize((args.input_size, args.input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),  
            ])
        else:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        
        dataset = datasets.ImageFolder(root=root, transform=transform)
        return dataset
    
    elif args.dataset == 'fusar_30per':
        print('Loading fusar_30per dataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/use/cls/New_FUSAR/FUSAR_30PER/Train'
        args.nb_classes = 10
        root = os.path.join(args.data_path, 'Train' if is_train else 'Val')
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        return dataset
    
    elif args.dataset == 'mstar':
        print('Loading mstar dataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/use/MSTAR/MSTAR_SOC/Train'
        args.nb_classes = 10
        root = os.path.join(args.data_path, 'Train' if is_train else 'Val')
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        return dataset

    elif args.dataset == 'mstar_10':
        print('Loading mstar_10 dataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/use/MSTAR/MSTAR_10/Train'
        args.nb_classes = 10
        root = os.path.join(args.data_path, 'Train' if is_train else 'Val')
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        return dataset
    
    elif args.dataset == 'mstar_20':
        print('Loading mstar_20 dataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/use/MSTAR/MSTAR_20/Train'
        args.nb_classes = 10
        root = os.path.join(args.data_path, 'Train' if is_train else 'Val')
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        return dataset
    
    elif args.dataset == 'mstar_40':
        print('Loading mstar_40 dataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/MSTAR_all_40/Train'
        args.nb_classes = 8
        root = os.path.join(args.data_path, 'Train' if is_train else 'Val')
        
        if is_train:
            transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
            ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        return dataset
    
    elif args.dataset == 'mstar_30per':
        print('Loading mstar_30per dataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/use/MSTAR/MSTAR_30PER/Train'
        args.nb_classes = 10
        root = os.path.join(args.data_path, 'Train' if is_train else 'Val')
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        return dataset
    
    elif args.dataset == 'mstar_all_30':
        print('Loading mstar_all_30 dataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/MSTAR_all_30/Train'
        args.nb_classes = 8
        root = os.path.join(args.data_path, 'Train' if is_train else 'Val')
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        return dataset
   
    elif args.dataset == 'rawsar':
        print('Loading RAWSARImageDataset!')
        data_path = '/data0/ldx/SAR_fdmodel/data/pretrain_sar/'
        dataset = INFRAREDDataset(data_path, train=is_train, transform=transform)
        args.nb_classes = 0       
    else:
        raise NotImplementedError

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    # if args.input_size <= 224:
    crop_pct = 224 / 256
    # else:
    #     crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
