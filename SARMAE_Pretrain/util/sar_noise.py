import torch
import numpy as np
from typing import Optional, Union, Tuple


class SARNoiseGenerator:
    
    def __init__(self, noise_std: float = 0.1, min_val: float = 1e-8, 
                 random_noise: bool = False, noise_range: tuple = (0.3, 0.7),
                 noise_ratio: float = 1.0):

        self.noise_std = noise_std
        self.min_val = min_val
        self.random_noise = random_noise
        self.noise_range = noise_range
        self.noise_ratio = noise_ratio
        
        if self.random_noise:
            print(f" 启用随机噪声强度: σ ∈ [{noise_range[0]:.2f}, {noise_range[1]:.2f}]")
        else:
            print(f" 固定噪声强度: σ = {noise_std:.2f}")
        
        if noise_ratio < 1.0:
            print(f" 噪声比例: {noise_ratio*100:.1f}% 的图像将被加噪")
    
    def add_multiplicative_noise(
        self, 
        image: torch.Tensor, 
        noise_std: Optional[float] = None
    ) -> torch.Tensor:

        if noise_std is None:
            if self.random_noise:

                min_std, max_std = self.noise_range
                noise_std = np.random.uniform(min_std, max_std)
            else:
                noise_std = self.noise_std

        image_pos = torch.clamp(image, min=self.min_val)

        if len(image_pos.shape) == 3 and image_pos.shape[0] == 3:

            channel_diff = torch.mean(torch.abs(image_pos[0] - image_pos[1])) + \
                          torch.mean(torch.abs(image_pos[0] - image_pos[2]))
            
            if channel_diff < 1e-6:  

                single_channel = image_pos[0:1]  # [1, H, W]
                
                if noise_std > 0.05:  

                    single_channel_safe = torch.clamp(single_channel, min=1e-8)
                    log_image = torch.log(single_channel_safe)

                    if torch.isinf(log_image).any():
                        print(f"️ Log变换产生inf，使用Gamma分布替代")

                        cv = noise_std * 2
                        enl = 1.0 / (cv * cv + 1e-8)
                        shape = enl
                        scale = 1.0 / enl
                        
                        gamma_noise = torch.distributions.Gamma(shape, 1.0/scale).sample(single_channel.shape)
                        gamma_noise = gamma_noise.to(single_channel.device)
                        noisy_single = single_channel * gamma_noise
                        mean_ratio = torch.mean(single_channel) / (torch.mean(noisy_single) + 1e-8)
                        noisy_single = noisy_single * mean_ratio
                    else:
                        noise = torch.randn_like(log_image) * noise_std
                        log_noisy = log_image + noise
                        noisy_single = torch.exp(log_noisy)

                        if torch.isnan(noisy_single).any() or torch.isinf(noisy_single).any():
                            print(f"️ Exp变换产生NaN/Inf，使用原图像")
                            noisy_single = single_channel
                else: 
                    cv = noise_std * 2
                    enl = 1.0 / (cv * cv + 1e-8)
                    shape = enl
                    scale = 1.0 / enl
                    
                    gamma_noise = torch.distributions.Gamma(shape, 1.0/scale).sample(single_channel.shape)
                    gamma_noise = gamma_noise.to(single_channel.device)
                    
                    noisy_single = single_channel * gamma_noise

                    mean_ratio = torch.mean(single_channel) / (torch.mean(noisy_single) + 1e-8)
                    noisy_single = noisy_single * mean_ratio

                noisy_image = noisy_single.repeat(3, 1, 1)
            else:

                if noise_std > 0.05:  
                    image_pos_safe = torch.clamp(image_pos, min=1e-8)
                    log_image = torch.log(image_pos_safe)

                    if torch.isinf(log_image).any():
                        print(f"️ 彩色图像Log变换产生inf，使用Gamma分布替代")

                        cv = noise_std * 2
                        enl = 1.0 / (cv * cv + 1e-8)
                        shape = enl
                        scale = 1.0 / enl
                        
                        gamma_noise = torch.distributions.Gamma(shape, 1.0/scale).sample(image_pos.shape)
                        gamma_noise = gamma_noise.to(image_pos.device)
                        noisy_image = image_pos * gamma_noise
                        mean_ratio = torch.mean(image_pos) / (torch.mean(noisy_image) + 1e-8)
                        noisy_image = noisy_image * mean_ratio
                    else:
                        noise = torch.randn_like(log_image) * noise_std
                        log_noisy = log_image + noise
                        noisy_image = torch.exp(log_noisy)

                        if torch.isnan(noisy_image).any() or torch.isinf(noisy_image).any():
                            print(f"️ 彩色图像Exp变换产生NaN/Inf，使用原图像")
                            noisy_image = image_pos
                else:
                    cv = noise_std * 2
                    enl = 1.0 / (cv * cv + 1e-8)
                    shape = enl
                    scale = 1.0 / enl
                    
                    gamma_noise = torch.distributions.Gamma(shape, 1.0/scale).sample(image_pos.shape)
                    gamma_noise = gamma_noise.to(image_pos.device)
                    
                    noisy_image = image_pos * gamma_noise
                    
                    mean_ratio = torch.mean(image_pos) / (torch.mean(noisy_image) + 1e-8)
                    noisy_image = noisy_image * mean_ratio
        else:

            if noise_std > 0.05:  
                image_pos_safe = torch.clamp(image_pos, min=1e-8)
                log_image = torch.log(image_pos_safe)

                if torch.isinf(log_image).any():
                    print(f"️ 其他形状图像Log变换产生inf，使用Gamma分布替代")

                    cv = noise_std * 2
                    enl = 1.0 / (cv * cv + 1e-8)
                    shape = enl
                    scale = 1.0 / enl
                    
                    gamma_noise = torch.distributions.Gamma(shape, 1.0/scale).sample(image_pos.shape)
                    gamma_noise = gamma_noise.to(image_pos.device)
                    noisy_image = image_pos * gamma_noise
                    mean_ratio = torch.mean(image_pos) / (torch.mean(noisy_image) + 1e-8)
                    noisy_image = noisy_image * mean_ratio
                else:
                    noise = torch.randn_like(log_image) * noise_std
                    log_noisy = log_image + noise
                    noisy_image = torch.exp(log_noisy)

                    if torch.isnan(noisy_image).any() or torch.isinf(noisy_image).any():
                        print(f"️ 其他形状图像Exp变换产生NaN/Inf，使用原图像")
                        noisy_image = image_pos
            else:
                cv = noise_std * 2
                enl = 1.0 / (cv * cv + 1e-8)
                shape = enl
                scale = 1.0 / enl
                
                gamma_noise = torch.distributions.Gamma(shape, 1.0/scale).sample(image_pos.shape)
                gamma_noise = gamma_noise.to(image_pos.device)
                
                noisy_image = image_pos * gamma_noise
                
                mean_ratio = torch.mean(image_pos) / (torch.mean(noisy_image) + 1e-8)
                noisy_image = noisy_image * mean_ratio

        noisy_image = torch.clamp(noisy_image, self.min_val, 1.0)

        if torch.isnan(noisy_image).any() or torch.isinf(noisy_image).any():
            print(f" 噪声添加后发现NaN/Inf，使用原图像")
            print(f"   - 噪声强度: {noise_std}")
            print(f"   - 原图统计: min={image.min():.6f}, max={image.max():.6f}")
            noisy_image = image  
        
        return noisy_image
    
    def add_noise_batch(
        self, 
        images: torch.Tensor, 
        noise_std: Optional[Union[float, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = images.size(0)
        device = images.device

        if self.noise_ratio < 1.0:
            noise_mask = torch.rand(batch_size, device=device) < self.noise_ratio
            num_noisy = noise_mask.sum().item()
        else:
            noise_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            num_noisy = batch_size
        
        if num_noisy == 0:
            return images.clone(), noise_mask
            
        if noise_std is None:
            noise_std = self.noise_std

        noisy_images = images.clone()

        if isinstance(noise_std, (int, float)):

            if num_noisy == batch_size:
                noisy_images = self.add_multiplicative_noise(images, noise_std)
            else:
                selected_images = images[noise_mask]
                noisy_selected = self.add_multiplicative_noise(selected_images, noise_std)
                noisy_images[noise_mask] = noisy_selected
        else:
            if isinstance(noise_std, torch.Tensor):
                assert noise_std.size(0) == batch_size, "noise_std长度必须与batch_size相同"

                noise_indices = torch.where(noise_mask)[0]
                for idx in noise_indices:
                    noisy_img = self.add_multiplicative_noise(
                        images[idx:idx+1], 
                        noise_std[idx].item()
                    )
                    noisy_images[idx] = noisy_img[0]
        
        return noisy_images, noise_mask
    
    def should_add_noise(self) -> bool:
        if self.noise_ratio >= 1.0:
            return True
        return torch.rand(1).item() < self.noise_ratio
    
    def set_noise_std(self, noise_std: float):
        self.noise_std = noise_std
    
    def get_noise_std(self) -> float:
        return self.noise_std


def create_sar_noise_scheduler(
    initial_std: float = 0.1,
    final_std: float = 0.05,
    total_epochs: int = 100,
    schedule_type: str = 'linear'
) -> callable:

    def scheduler(epoch: int) -> float:
        progress = epoch / max(total_epochs - 1, 1)
        
        if schedule_type == 'linear':
            noise_std = initial_std + (final_std - initial_std) * progress
        elif schedule_type == 'cosine':
            noise_std = final_std + (initial_std - final_std) * 0.5 * (1 + np.cos(np.pi * progress))
        elif schedule_type == 'exponential':
            decay_rate = np.log(final_std / initial_std)
            noise_std = initial_std * np.exp(decay_rate * progress)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")
        
        return max(noise_std, 0.01)  
    
    return scheduler


def add_sar_multiplicative_noise(
    image: torch.Tensor, 
    noise_std: float = 0.1,
    min_val: float = 1e-8
) -> torch.Tensor:

    generator = SARNoiseGenerator(noise_std, min_val)
    return generator.add_multiplicative_noise(image)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    test_image = torch.rand(1, 3, 224, 224) * 0.8 + 0.1  

    noise_generator = SARNoiseGenerator()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(test_image[0, 0].numpy(), cmap='gray')
    axes[0, 0].set_title('Original')

    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
    for i, std in enumerate(noise_levels):
        noisy = noise_generator.add_multiplicative_noise(test_image, std)
        row = i // 3
        col = (i + 1) % 3
        if row < 2 and col < 3:
            axes[row, col].imshow(noisy[0, 0].numpy(), cmap='gray')
            axes[row, col].set_title(f'Noise std={std}')
    
    plt.tight_layout()
    plt.savefig('sar_noise_test.png')
    print(" SAR噪声测试完成，结果保存为 sar_noise_test.png")
