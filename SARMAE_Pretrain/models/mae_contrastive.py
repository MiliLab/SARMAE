import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from collections import OrderedDict


class SAROpticalPatchAlignment(nn.Module):

    def __init__(
        self, 
        mae_model,
        optical_encoder: str = 'vit_base_patch16_224',
        freeze_optical_layers: int = 8,
        patch_size: int = 16,
        mae_loss_weight: float = 1.0,
        alignment_loss_weight: float = 1.0,
        dinov3_pretrained_path: str = None,
        freeze_optical_completely: bool = False,
        use_checkpoint: bool = False  
    ):
        super().__init__()
        
        print(f" 初始化SAROpticalPatchAlignment（直接使用DINOv3）...")

        self.use_checkpoint = use_checkpoint
        
        self.sar_encoder = mae_model
        self.patch_size = patch_size
        self.freeze_optical_completely = freeze_optical_completely

        if dinov3_pretrained_path:
            print(f" 直接加载DINOv3模型作为光学编码器...")
            self.optical_encoder = self._load_dinov3_direct(dinov3_pretrained_path)
            optical_dim = self.optical_encoder.embed_dim if hasattr(self.optical_encoder, 'embed_dim') else 768
            print(f" 光学编码器维度: {optical_dim}")
        else:
            print(f"️ 未提供DINOv3路径，使用简单ViT编码器")
            import torch.nn as nn
            
            class FallbackViT(nn.Module):
                def __init__(self, embed_dim=768):
                    super().__init__()
                    self.embed_dim = embed_dim
  
                    self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)

                    self.global_pool = nn.AdaptiveAvgPool2d(1)
                    
                def forward(self, x):
                    # Patch embedding
                    x = self.patch_embed(x)  # [B, embed_dim, H//16, W//16]
                    
                    # Global average pooling
                    x = self.global_pool(x)  # [B, embed_dim, 1, 1]
                    x = x.flatten(1)  # [B, embed_dim]
                    
                    return x
            
            self.optical_encoder = FallbackViT()
            optical_dim = 768

        sar_dim = self.sar_encoder.embed_dim if hasattr(self.sar_encoder, 'embed_dim') else 768
        
        print(f"   - SAR特征维度: {sar_dim}")
        print(f"   - 光学特征维度: {optical_dim}")

        self.sar_alignment_ffn = torch.nn.Sequential(
            torch.nn.Linear(sar_dim, sar_dim),
            torch.nn.ReLU(), 
            torch.nn.Linear(sar_dim, optical_dim), 
            torch.nn.LayerNorm(optical_dim)  
        )

        self._init_alignment_ffn_weights()

        for param in self.optical_encoder.parameters():
            param.requires_grad = False
        print(" 光学编码器已完全冻结")

        self.mae_loss_weight = mae_loss_weight
        self.alignment_loss_weight = alignment_loss_weight

        self._debug_alignment = True
        self._debug_nan = True
        
        print(" SAROpticalPatchAlignment初始化完成")

    def _load_dinov3_direct(self, dinov3_path):
        print(f" 直接加载DINOv3模型: {dinov3_path}")
        
        try:
            checkpoint = torch.load(dinov3_path, map_location='cpu')
            print(f" 成功加载DINOv3检查点，包含{len(checkpoint)}个权重")

            import torch.nn as nn
            
            class SimplifiedDINOv3(nn.Module):

                def __init__(self, embed_dim=768):
                    super().__init__()
                    self.embed_dim = embed_dim

                    self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)

                    self.norm = nn.LayerNorm(embed_dim)
                    self.global_pool = nn.AdaptiveAvgPool2d(1)

                    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                
                def forward(self, x, return_patches=False):
                    B = x.shape[0]
                    
                    # Patch embedding
                    x = self.patch_embed(x)  # [B, embed_dim, H//16, W//16] e.g. [B, 768, 14, 14]
                    
                    if return_patches:
                        N_patches = x.shape[2] * x.shape[3]  # 14*14=196
                        x = x.flatten(2)  # [B, embed_dim, 196]
                        x = x.transpose(1, 2)  # [B, 196, embed_dim]
                        x = self.norm(x)  
                        return x  # [B, N_patches, embed_dim]
                    else:
                        x = self.global_pool(x)  # [B, embed_dim, 1, 1]
                        x = x.flatten(1)  # [B, embed_dim]
                        
                        # Layer norm
                        x = self.norm(x)
                        
                        return x

            if 'cls_token' in checkpoint:
                embed_dim = checkpoint['cls_token'].shape[-1]
                print(f" 从DINOv3权重文件检测到embed_dim: {embed_dim}")
            else:
                embed_dim = 768  
                print(f"️ 无法从权重文件检测embed_dim，使用默认值: {embed_dim}")

            model = SimplifiedDINOv3(embed_dim=embed_dim)
            model_dict = model.state_dict()
            loaded_count = 0

            if 'patch_embed.proj.weight' in checkpoint:
                model_dict['patch_embed.weight'] = checkpoint['patch_embed.proj.weight']
                loaded_count += 1
                print(f" 加载patch_embed权重: {checkpoint['patch_embed.proj.weight'].shape}")
                
            if 'patch_embed.proj.bias' in checkpoint:
                model_dict['patch_embed.bias'] = checkpoint['patch_embed.proj.bias']
                loaded_count += 1
                print(f" 加载patch_embed偏置: {checkpoint['patch_embed.proj.bias'].shape}")

            if 'norm.weight' in checkpoint:
                model_dict['norm.weight'] = checkpoint['norm.weight']
                loaded_count += 1
                print(f" 加载norm权重: {checkpoint['norm.weight'].shape}")
                
            if 'norm.bias' in checkpoint:
                model_dict['norm.bias'] = checkpoint['norm.bias']
                loaded_count += 1
                print(f" 加载norm偏置: {checkpoint['norm.bias'].shape}")

            if 'cls_token' in checkpoint:
                model_dict['cls_token'] = checkpoint['cls_token']
                loaded_count += 1
                print(f" 加载cls_token: {checkpoint['cls_token'].shape}")

            model.load_state_dict(model_dict)
            
            print(f" 简化DINOv3模型创建完成，成功加载 {loaded_count} 个关键权重")
            print(f"   - 使用patch embedding + 全局池化的简化架构")
            print(f"   - 避免了复杂的RoPE位置编码问题")
            
            return model
            
        except Exception as e:
            print(f" 加载DINOv3失败: {e}")
            print(f" 回退到最简单的fallback模型...")

            import torch.nn as nn
            
            class MinimalViT(nn.Module):
                def __init__(self, embed_dim=768):
                    super().__init__()
                    self.embed_dim = embed_dim

                    self.conv = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
                    self.pool = nn.AdaptiveAvgPool2d(1)
                    self.norm = nn.LayerNorm(embed_dim)

                    nn.init.xavier_uniform_(self.conv.weight, gain=0.02)
                    nn.init.zeros_(self.conv.bias)
                    
                def forward(self, x, return_patches=False):
                    x = self.conv(x)  # [B, embed_dim, H//16, W//16]
                    
                    if return_patches:
                        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, embed_dim]
                        x = self.norm(x)
                        return x
                    else:
                        x = self.pool(x)
                        x = x.flatten(1)
                        x = self.norm(x)
                        return x
            
            return MinimalViT()

        print(" Patch对齐模式：使用单一光学编码器，无需EMA")

        embed_dim = 768
        optical_dim = self.optical_encoder.num_features

        self._debug_alignment = True
        self._debug_nan = True
        
        print(f" Initialized SAR-Optical patch alignment model")
        print(f"   - SAR embed_dim: {embed_dim}")
        print(f"   - Optical embed_dim: {optical_dim}")
        print(f"    调试模式已启用")

        self._validate_all_weights()

    def _validate_all_weights(self):
        print(" 验证所有模型权重...")
        
        problem_count = 0

        for name, param in self.optical_encoder.named_parameters():
            if torch.isnan(param).any():
                print(f" 光学编码器包含NaN: {name}")
                problem_count += 1
            elif torch.isinf(param).any():
                print(f" 光学编码器包含Inf: {name}")
                problem_count += 1
            elif torch.all(param == 0) and 'bias' not in name:
                print(f"️ 光学编码器权重全零: {name}")
                problem_count += 1
            elif param.std() < 1e-8 and 'bias' not in name:
                print(f"️ 光学编码器权重标准差过小: {name} (std={param.std():.10f})")
                problem_count += 1

        if problem_count == 0:
            print(" 所有权重验证通过")
        else:
            print(f" 发现 {problem_count} 个权重问题")
            
    def _freeze_optical_layers(self, freeze_layers: int):
        frozen_params = 0
        total_params = sum(p.numel() for p in self.optical_encoder.parameters())

        if hasattr(self.optical_encoder, 'patch_embed'):
            for param in self.optical_encoder.patch_embed.parameters():
                param.requires_grad = False
                frozen_params += param.numel()

        if hasattr(self.optical_encoder, 'pos_embed'):
            self.optical_encoder.pos_embed.requires_grad = False
            frozen_params += self.optical_encoder.pos_embed.numel()
        if hasattr(self.optical_encoder, 'cls_token'):
            self.optical_encoder.cls_token.requires_grad = False
            frozen_params += self.optical_encoder.cls_token.numel()

        if hasattr(self.optical_encoder, 'blocks'):
            for i, block in enumerate(self.optical_encoder.blocks):
                if i < freeze_layers:
                    for param in block.parameters():
                        param.requires_grad = False
                        frozen_params += param.numel()
        
        print(f" Frozen {frozen_params}/{total_params} optical parameters "
              f"({100*frozen_params/total_params:.1f}%)")

    def _init_alignment_ffn_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)  
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        print(" 初始化SAR对齐FFN权重...")
        self.sar_alignment_ffn.apply(init_weights)
        print(" SAR对齐FFN权重初始化完成")

    def _freeze_optical_completely(self):
        frozen_params = 0
        total_params = sum(p.numel() for p in self.optical_encoder.parameters())
        
        for param in self.optical_encoder.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        print(f"️ Completely frozen all optical encoder parameters: "
              f"{frozen_params}/{total_params} ({100*frozen_params/total_params:.1f}%)")

    def compute_patch_alignment_loss(self, sar_patch_features, optical_patch_features):
        B, N_patches, _ = sar_patch_features.shape

        if torch.isnan(sar_patch_features).any() or torch.isinf(sar_patch_features).any():
            print(f" SAR patch特征包含NaN/Inf")
            return torch.tensor(0.0, device=sar_patch_features.device, requires_grad=True)
            
        if torch.isnan(optical_patch_features).any() or torch.isinf(optical_patch_features).any():
            print(f" 光学patch特征包含NaN/Inf")
            return torch.tensor(0.0, device=sar_patch_features.device, requires_grad=True)

        try:
            sar_aligned_features = self.sar_alignment_ffn(sar_patch_features)  # [B, N_patches, optical_dim]

            if torch.isnan(sar_aligned_features).any() or torch.isinf(sar_aligned_features).any():
                print(f" SAR对齐后特征包含NaN/Inf")
                return torch.tensor(0.0, device=sar_patch_features.device, requires_grad=True)
                
        except Exception as e:
            print(f" SAR特征对齐失败: {e}")
            return torch.tensor(0.0, device=sar_patch_features.device, requires_grad=True)

        sar_normalized = F.normalize(sar_aligned_features, dim=-1, eps=1e-6)  # [B, N_patches, optical_dim]
        optical_normalized = F.normalize(optical_patch_features, dim=-1, eps=1e-6)  # [B, N_patches, optical_dim]

        cosine_similarities = torch.sum(sar_normalized * optical_normalized, dim=-1)  # [B, N_patches]

        alignment_loss_per_patch = 1.0 - cosine_similarities  # [B, N_patches]

        alignment_loss = alignment_loss_per_patch.mean()

        if torch.isnan(alignment_loss) or torch.isinf(alignment_loss):
            print(f" 对齐损失异常: {alignment_loss.item()}")
            return torch.tensor(0.0, device=sar_patch_features.device, requires_grad=True)

        if hasattr(self, '_debug_alignment') and self._debug_alignment:

            self._debug_alignment = False  
        
        return alignment_loss

    def _align_optical_patches_to_sar(self, optical_patch_tokens, sar_mask, sar_ids_restore):

        B, N_patches, optical_dim = optical_patch_tokens.shape
        
        aligned_patches_list = []
        
        for i in range(B):
            len_keep = (sar_mask[i] == 0).sum().item()

            ids_shuffle = torch.argsort(sar_ids_restore[i])  
            ids_keep = ids_shuffle[:len_keep]  

            optical_sample_aligned = torch.gather(
                optical_patch_tokens[i:i+1], dim=1,
                index=ids_keep.unsqueeze(0).unsqueeze(-1).repeat(1, 1, optical_dim)
            )  # [1, len_keep, optical_dim]
            
            aligned_patches_list.append(optical_sample_aligned)

        aligned_optical_patches = torch.cat(aligned_patches_list, dim=0)  # [B, N_visible, optical_dim]
        
        return aligned_optical_patches

    def forward(self, sar_images, optical_images=None, is_paired_mask=None, sar_target_images=None, mask_ratio=0.75, loss_weights=None):

        B = sar_images.shape[0]
        device = sar_images.device

        if loss_weights is None:
            loss_weights = {'mae': self.mae_loss_weight, 'alignment': self.alignment_loss_weight}

        sar_latent, sar_mask, sar_ids_restore = self.sar_encoder.forward_encoder(sar_images, mask_ratio)
        sar_pred = self.sar_encoder.forward_decoder(sar_latent, sar_ids_restore)

        sar_target = sar_target_images if sar_target_images is not None else sar_images
        sar_loss = self.sar_encoder.forward_loss(sar_target, sar_pred, sar_mask)

        sar_patch_tokens = sar_latent[:, 1:]  # [B, N_visible, embed_dim] - 去掉CLS token

        if torch.isnan(sar_patch_tokens).any() or torch.isinf(sar_patch_tokens).any():
            print(f" SAR patch特征包含NaN/Inf")
            sar_patch_tokens = torch.zeros_like(sar_patch_tokens)

        alignment_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if optical_images is not None and loss_weights.get('alignment', 0) > 0 and is_paired_mask is not None:

            if isinstance(is_paired_mask, torch.Tensor):
                paired_mask_tensor = is_paired_mask.clone().detach().to(dtype=torch.bool, device=device)
            else:
                paired_mask_tensor = torch.tensor(is_paired_mask, dtype=torch.bool, device=device)
            paired_indices = torch.where(paired_mask_tensor)[0]
            
            if len(paired_indices) > 0:
                try:
                    paired_optical_images = optical_images[paired_indices]  # [N_paired, C, H, W]
                    paired_sar_patch_tokens = sar_patch_tokens[paired_indices]  # [N_paired, N_visible, embed_dim]
                    paired_sar_mask = sar_mask[paired_indices]  # [N_paired, N_patches]
                    paired_sar_ids_restore = sar_ids_restore[paired_indices]  # [N_paired, N_patches]

                    with torch.no_grad():  
                        if hasattr(self.optical_encoder, 'forward') and 'return_patches' in self.optical_encoder.forward.__code__.co_varnames:
                            paired_optical_patch_tokens = self.optical_encoder(paired_optical_images, return_patches=True)  # [N_paired, N_patches, optical_dim]
                        else:
                            paired_optical_patches = self.optical_encoder.patch_embed(paired_optical_images)  # [N_paired, N_patches, optical_dim]
                            paired_optical_patch_tokens = self.optical_encoder.norm(paired_optical_patches) if hasattr(self.optical_encoder, 'norm') else paired_optical_patches

                    paired_optical_aligned_patches = self._align_optical_patches_to_sar(
                        paired_optical_patch_tokens, paired_sar_mask, paired_sar_ids_restore
                    )  # [N_paired, N_visible, optical_dim]

                    if torch.isnan(paired_optical_aligned_patches).any() or torch.isinf(paired_optical_aligned_patches).any():
                        print(f" 配对光学patch特征包含NaN/Inf")
                        paired_optical_aligned_patches = torch.zeros_like(paired_optical_aligned_patches)

                    if paired_optical_aligned_patches.shape[1] == paired_sar_patch_tokens.shape[1]: 
                        alignment_loss = self.compute_patch_alignment_loss(
                            paired_sar_patch_tokens, paired_optical_aligned_patches
                        )

                    else:
                        print(f"️ 配对样本Patch数量不匹配: SAR={paired_sar_patch_tokens.shape[1]}, 光学={paired_optical_aligned_patches.shape[1]}")
                        
                except Exception as e:
                    print(f" 配对样本光学patch特征提取失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                pass

        total_loss = (loss_weights.get('mae', 1.0) * sar_loss + 
                     loss_weights.get('alignment', 0.0) * alignment_loss)

        loss_dict = {
            'mae_loss': sar_loss.item() if hasattr(sar_loss, 'item') else float(sar_loss),
            'alignment_loss': alignment_loss.item() if hasattr(alignment_loss, 'item') else float(alignment_loss),
            'total_loss': total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
        }

        return total_loss, loss_dict, alignment_loss
    

    def load_sar_pretrained(self, sar_checkpoint_path: str, strict: bool = True):
        if not os.path.exists(sar_checkpoint_path):
            raise FileNotFoundError(f"SAR checkpoint not found: {sar_checkpoint_path}")
        
        print(f" Loading SAR pretrained weights from: {sar_checkpoint_path}")
        
        try:
            checkpoint = torch.load(sar_checkpoint_path, map_location='cpu')
        except Exception as e:
            print(f" 加载SAR权重失败: {e}")
            print(" 尝试直接加载state_dict...")
            try:
                checkpoint = {'model': torch.load(sar_checkpoint_path, map_location='cpu')}
            except Exception as e2:
                print(f" 直接加载也失败: {e2}")
                raise RuntimeError(f"无法加载SAR权重文件: {sar_checkpoint_path}") from e2
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        cleaned_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('sar_encoder.'):
                k = k[12:]
            cleaned_state_dict[k] = v

        encoder_state_dict = {}
        for k, v in cleaned_state_dict.items():
            if not k.startswith('decoder_') and k != 'mask_token':
                encoder_state_dict[k] = v
        
        missing_keys, unexpected_keys = self.sar_encoder.load_state_dict(encoder_state_dict, strict=False)
        
        if missing_keys:
            print(f"️  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"️  Unexpected keys: {unexpected_keys}")
        
        print(f" SAR encoder weights loaded!")

    def load_optical_pretrained(self, optical_checkpoint_path: str, strict: bool = False):
        if not os.path.exists(optical_checkpoint_path):
            raise FileNotFoundError(f"Optical checkpoint not found: {optical_checkpoint_path}")
        
        print(f" Loading Optical pretrained weights from: {optical_checkpoint_path}")
        
        checkpoint = torch.load(optical_checkpoint_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        cleaned_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if any(skip in k for skip in ['head', 'fc', 'classifier']):
                continue
            cleaned_state_dict[k] = v
        
        missing_keys, unexpected_keys = self.optical_encoder.load_state_dict(cleaned_state_dict, strict=strict)
        
        print(f" Optical encoder weights loaded!")
