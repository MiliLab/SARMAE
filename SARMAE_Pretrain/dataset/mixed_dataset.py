import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from util.sar_noise import SARNoiseGenerator


class MixedSARDataset(Dataset):

    def __init__(self, sar_dir, optical_dir, paired_json, unpaired_json, 
                 transform_sar=None, transform_optical=None, paired_ratio=0.7,
                 enable_sar_noise=False, noise_std=0.1, 
                 random_noise=False, noise_range=(0.3, 0.7), noise_ratio=1.0):

        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.transform_sar = transform_sar
        self.transform_optical = transform_optical
        self.paired_ratio = paired_ratio
        self.enable_sar_noise = enable_sar_noise
        self.noise_std = noise_std
        self.random_noise = random_noise
        self.noise_range = noise_range
        self.noise_ratio = noise_ratio

        if self.enable_sar_noise:
            self.noise_generator = SARNoiseGenerator(
                noise_std=noise_std,
                random_noise=random_noise,
                noise_range=noise_range,
                noise_ratio=noise_ratio
            )
        else:
            self.noise_generator = None

        self._load_data(paired_json, unpaired_json)
        
        print(f" 数据加载完成:")
        print(f"   - 配对样本: {len(self.paired_data)}")
        print(f"   - 非配对样本: {len(self.unpaired_data)}")
        print(f"   - 数据集总长度: {len(self)}")
        print(f"   - 配对比例: {self.paired_ratio:.1%}")
    
    def _load_data(self, paired_json, unpaired_json):

        print(" 加载配对数据...")
        with open(paired_json, 'r') as f:
            self.paired_data = json.load(f)
        
        print(" 加载非配对数据...")
        with open(unpaired_json, 'r') as f:
            self.unpaired_data = json.load(f)
    
    def __len__(self):

        if len(self.paired_data) > 0:
            # 根据配对数据量和比例计算总样本数
            total_samples = int(len(self.paired_data) / self.paired_ratio)
        else:
            total_samples = len(self.unpaired_data)
        
        return total_samples
    
    def __getitem__(self, idx):

        total_length = len(self)
        paired_cutoff = int(total_length * self.paired_ratio)
        
        if idx < paired_cutoff:
            return self._get_paired_sample(idx)
        else:
            unpaired_idx = idx - paired_cutoff
            return self._get_unpaired_sample(unpaired_idx)
    
    def _get_paired_sample(self, idx):

        try:
            paired_idx = idx % len(self.paired_data)
            paired_info = self.paired_data[paired_idx]

            sar_filename = paired_info['sar']
            if sar_filename.startswith('SAR/'):
                sar_filename = sar_filename[4:]  
        
            optical_filename = paired_info['optical']
            if optical_filename.startswith('OPT/'):
                optical_filename = optical_filename[4:]  

            sar_path = os.path.join(self.sar_dir, sar_filename)
            optical_path = os.path.join(self.optical_dir, optical_filename)

            if not os.path.exists(sar_path):
                print(f"️ SAR文件不存在: {sar_path}")
                return self._get_fallback_paired_sample()
            
            if not os.path.exists(optical_path):
                print(f"️ 光学文件不存在: {optical_path}")
                return self._get_fallback_paired_sample()

            sar_img = Image.open(sar_path)  
            optical_img = Image.open(optical_path).convert('RGB')

            if self.enable_sar_noise and self.noise_generator:
                sar_tensor_01 = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])(sar_img)

                if sar_tensor_01.shape[0] == 1:
                    sar_tensor_01 = sar_tensor_01.repeat(3, 1, 1)

                sar_clean_01 = sar_tensor_01.clone()
                if self.noise_generator.should_add_noise():
                    sar_noisy_01 = self.noise_generator.add_multiplicative_noise(sar_tensor_01)
                else:
                    sar_noisy_01 = sar_clean_01.clone()  

                normalize_transform = transforms.Normalize([0.5] * 3, [0.5] * 3)
                sar_clean = normalize_transform(sar_clean_01)  
                sar_noisy = normalize_transform(sar_noisy_01)  

                if self.transform_optical:
                    optical_tensor = self.transform_optical(optical_img)
                else:
                    optical_tensor = torch.from_numpy(np.array(optical_img)).permute(2, 0, 1).float() / 255.0
                
                return {
                    'sar': sar_noisy,        
                    'sar_target': sar_clean, 
                    'optical': optical_tensor,
                    'is_paired': True
                }
            else:

                if self.transform_sar:
                    sar_tensor = self.transform_sar(sar_img)
                else:
                    sar_tensor = torch.from_numpy(np.array(sar_img)).unsqueeze(0).float() / 255.0
                
                if self.transform_optical:
                    optical_tensor = self.transform_optical(optical_img)
                else:
                    optical_tensor = torch.from_numpy(np.array(optical_img)).permute(2, 0, 1).float() / 255.0
                
                return {
                    'sar': sar_tensor,
                    'sar_target': sar_tensor,  
                    'optical': optical_tensor,
                    'is_paired': True
                }
            
        except Exception as e:
            print(f" 加载配对样本失败 idx={idx}: {e}")
            return self._get_fallback_paired_sample()
    
    def _get_unpaired_sample(self, unpaired_idx):

        try:
            real_idx = unpaired_idx % len(self.unpaired_data)
            unpaired_info = self.unpaired_data[real_idx]

            if isinstance(unpaired_info, dict):
                sar_filename = unpaired_info['sar']
            else:
                sar_filename = unpaired_info  
            
            sar_path = os.path.join(self.sar_dir, sar_filename)

            if not os.path.exists(sar_path):
                print(f"️ SAR文件不存在: {sar_path}")
                return self._get_fallback_unpaired_sample()

            sar_img = Image.open(sar_path)  

            if self.enable_sar_noise and self.noise_generator:

                sar_tensor_01 = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])(sar_img)

                if sar_tensor_01.shape[0] == 1:
                    sar_tensor_01 = sar_tensor_01.repeat(3, 1, 1)

                sar_clean_01 = sar_tensor_01.clone()
                if self.noise_generator.should_add_noise():
                    sar_noisy_01 = self.noise_generator.add_multiplicative_noise(sar_tensor_01)
                else:
                    sar_noisy_01 = sar_clean_01.clone()  

                normalize_transform = transforms.Normalize([0.5] * 3, [0.5] * 3)
                sar_clean = normalize_transform(sar_clean_01)  
                sar_noisy = normalize_transform(sar_noisy_01)  
                
                return {
                    'sar': sar_noisy,        
                    'sar_target': sar_clean, 
                    'is_paired': False
                }
            else:
                if self.transform_sar:
                    sar_tensor = self.transform_sar(sar_img)
                else:
                    sar_tensor = torch.from_numpy(np.array(sar_img)).unsqueeze(0).float() / 255.0
                
                return {
                    'sar': sar_tensor,
                    'sar_target': sar_tensor,  
                    'is_paired': False
                }
            
        except Exception as e:
            print(f" 加载非配对样本失败 idx={unpaired_idx}: {e}")
            return self._get_fallback_unpaired_sample()
    
    def _get_fallback_paired_sample(self):
        """fallback配对样本（当加载失败时使用）"""
        sar_tensor = torch.zeros(3, 224, 224)
        return {
            'sar': sar_tensor,                     
            'sar_target': sar_tensor.clone(),       
            'optical': torch.zeros(3, 224, 224),    
            'is_paired': True
        }
    
    def _get_fallback_unpaired_sample(self):
        """fallback非配对样本（当加载失败时使用）"""
        sar_tensor = torch.zeros(3, 224, 224)
        return {
            'sar': sar_tensor,                      
            'sar_target': sar_tensor.clone(),       
            'is_paired': False
        }
    
    def get_sample_info(self, idx):
        """获取样本信息（用于调试）"""
        total_length = len(self)
        paired_cutoff = int(total_length * self.paired_ratio)
        
        if idx < paired_cutoff:
            paired_idx = idx % len(self.paired_data)
            return f"配对样本 {idx} -> paired_data[{paired_idx}]"
        else:
            unpaired_idx = (idx - paired_cutoff) % len(self.unpaired_data)
            return f"非配对样本 {idx} -> unpaired_data[{unpaired_idx}]"

def mixed_collate_fn(batch):
    """
    自定义collate函数，处理有些样本有optical，有些没有的情况
    """
    sar_list = []
    optical_list = []
    is_paired_list = []
    
    for sample in batch:
        sar_list.append(sample['sar'])
        is_paired_list.append(sample['is_paired'])
        
        if sample['is_paired']:
            optical_list.append(sample['optical'])

    result = {
        'sar': torch.stack(sar_list),
        'is_paired': is_paired_list
    }

    if optical_list:
        result['optical'] = torch.stack(optical_list)
    
    return result


def create_mixed_dataset(data_path, input_size=224, paired_ratio=0.7):

    from dataset.transforms import build_sar_transform, build_optical_transform, RepeatChannels

    sar_dir = os.path.join(data_path, 'SAR')
    optical_dir = os.path.join(data_path, 'OPT')
    paired_json = os.path.join(data_path, 'paired.json')
    unpaired_json = os.path.join(data_path, 'unpaired.json')

    transform_sar = build_sar_transform(size=input_size)
    transform_optical = build_optical_transform(size=input_size)

    dataset = MixedSARDataset(
        sar_dir=sar_dir,
        optical_dir=optical_dir,
        paired_json=paired_json,
        unpaired_json=unpaired_json,
        transform_sar=transform_sar,
        transform_optical=transform_optical,
        paired_ratio=paired_ratio
    )
    
    return dataset


if __name__ == "__main__":
    import numpy as np
    
    dataset = create_mixed_dataset("/path/to/your/data")
    
    print(f"数据集长度: {len(dataset)}")

    for i in [0, 100, 500, 1000]:
        if i < len(dataset):
            print(f"\n样本 {i}: {dataset.get_sample_info(i)}")
            sample = dataset[i]
            print(f"  - SAR shape: {sample['sar'].shape}")
            print(f"  - is_paired: {sample['is_paired']}")
            if 'optical' in sample:
                print(f"  - Optical shape: {sample['optical'].shape}")