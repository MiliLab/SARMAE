import os
import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import cv2
    PLOTTING_AVAILABLE = True

    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        pass
    
    sns.set_palette("husl")
except ImportError as e:
    PLOTTING_AVAILABLE = False
    print(f"️ 可视化依赖不可用: {e}")

    class DummyModule:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    plt = DummyModule()
    sns = DummyModule()
    Image = DummyModule()
    TSNE = DummyModule()
    PCA = DummyModule()
    cv2 = DummyModule()


class MAEVisualizer:
    
    def __init__(self, save_dir="./visualizations"):

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'reconstructions'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'features'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'attention'), exist_ok=True)
        
        print(f" 可视化工具初始化完成，保存目录: {save_dir}")
    
    def visualize_masks(self, images, masks, ids_restore, epoch, batch_idx, 
                       image_type="SAR", num_samples=4, patch_size=16):

        device = images.device
        B, C, H, W = images.shape

        num_samples = min(num_samples, B)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):

            img = images[i].cpu()

            if image_type == "OPT":
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean  
            else:
                img = (img + 1.0) / 2.0
            
            if C == 3:
                img_np = img.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
            else:
                img_np = img.squeeze().numpy()
                img_np = np.clip(img_np, 0, 1)
            
            axes[i, 0].imshow(img_np, cmap='gray' if C == 1 else None)
            axes[i, 0].set_title(f'{image_type} Original')
            axes[i, 0].axis('off')

            mask = masks[i].cpu().numpy()  # [N]
            h = w = H // patch_size

            mask_img = mask.reshape(h, w)
            mask_img_resized = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
            
            axes[i, 1].imshow(mask_img_resized, cmap='RdYlBu', alpha=0.7)
            axes[i, 1].set_title(f'Mask (ratio={mask.mean():.2f})')
            axes[i, 1].axis('off')

            masked_img = img_np.copy()
            if C == 3:
                masked_img[mask_img_resized == 1] = [0.5, 0.5, 0.5]  
            else:
                masked_img[mask_img_resized == 1] = 0.5
            
            axes[i, 2].imshow(masked_img, cmap='gray' if C == 1 else None)
            axes[i, 2].set_title(f'{image_type} Masked')
            axes[i, 2].axis('off')
        
        plt.suptitle(f'{image_type} Mask Visualization - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'masks', 
                                f'{image_type}_masks_epoch{epoch:04d}_batch{batch_idx:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" 掩码可视化已保存: {save_path}")
    
    def visualize_reconstruction(self, original, reconstructed, masks, epoch, batch_idx,
                               image_type="SAR", num_samples=4, patch_size=16):

        device = original.device
        B, C, H, W = original.shape
        num_samples = min(num_samples, B)

        reconstructed_imgs = self._mae_reconstruct_with_mask(original, reconstructed, masks, H, W, C, patch_size)
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            orig_img = original[i].cpu()

            orig_img = (orig_img + 1.0) / 2.0
            
            if C == 3:
                orig_np = orig_img.permute(1, 2, 0).numpy()
                orig_np = np.clip(orig_np, 0, 1)
            else:
                orig_np = orig_img.squeeze().numpy()
                orig_np = np.clip(orig_np, 0, 1)
            
            axes[i, 0].imshow(orig_np, cmap='gray' if C == 1 else None)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')

            recon_img = reconstructed_imgs[i].cpu()

            recon_img = (recon_img + 1.0) / 2.0
            
            if C == 3:
                recon_np = recon_img.permute(1, 2, 0).numpy()
                recon_np = np.clip(recon_np, 0, 1)
            else:
                recon_np = recon_img.squeeze().numpy()
                recon_np = np.clip(recon_np, 0, 1)
            
            axes[i, 1].imshow(recon_np, cmap='gray' if C == 1 else None)
            axes[i, 1].set_title('Reconstructed')
            axes[i, 1].axis('off')

            if C == 3:
                diff = np.abs(orig_np - recon_np).mean(axis=2)
            else:
                diff = np.abs(orig_np - recon_np)
            
            im = axes[i, 2].imshow(diff, cmap='hot')
            axes[i, 2].set_title('Difference')
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

            mask = masks[i].cpu().numpy()
            h = w = H // patch_size
            mask_img = mask.reshape(h, w)
            mask_img_resized = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
            
            recon_only = orig_np.copy()
            if C == 3:
                recon_only[mask_img_resized == 1] = recon_np[mask_img_resized == 1]
            else:
                recon_only[mask_img_resized == 1] = recon_np[mask_img_resized == 1]
            
            axes[i, 3].imshow(recon_only, cmap='gray' if C == 1 else None)
            axes[i, 3].set_title('Reconstruction Only')
            axes[i, 3].axis('off')
        
        plt.suptitle(f'{image_type} Reconstruction - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'reconstructions',
                                f'{image_type}_recon_epoch{epoch:04d}_batch{batch_idx:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" 重建可视化已保存: {save_path}")
    
    def visualize_features(self, sar_features, optical_features, labels, epoch, 
                          method='tsne', num_samples=1000):

        sar_feat = sar_features.detach().cpu().numpy()
        opt_feat = optical_features.detach().cpu().numpy()
        labels_np = np.array(labels)

        if len(sar_feat) > num_samples:
            indices = np.random.choice(len(sar_feat), num_samples, replace=False)
            sar_feat = sar_feat[indices]
            opt_feat = opt_feat[indices]
            labels_np = labels_np[indices]

        all_features = np.vstack([sar_feat, opt_feat])
        all_labels = np.hstack([
            np.full(len(sar_feat), 0),  
            np.full(len(opt_feat), 1)   
        ])

        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=2, random_state=42)
        
        features_2d = reducer.fit_transform(all_features)

        sar_2d = features_2d[:len(sar_feat)]
        opt_2d = features_2d[len(sar_feat):]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].scatter(sar_2d[:, 0], sar_2d[:, 1], c='red', alpha=0.6, s=20, label='SAR')
        axes[0].scatter(opt_2d[:, 0], opt_2d[:, 1], c='blue', alpha=0.6, s=20, label='OPT')
        axes[0].set_title(f'Feature Distribution ({method.upper()}) - Epoch {epoch}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        paired_indices = labels_np[:len(sar_feat)] == True
        unpaired_indices = labels_np[:len(sar_feat)] == False
        
        if np.any(paired_indices):
            axes[1].scatter(sar_2d[paired_indices, 0], sar_2d[paired_indices, 1], 
                           c='darkred', alpha=0.8, s=30, label='SAR (Paired)')
            axes[1].scatter(opt_2d[paired_indices, 0], opt_2d[paired_indices, 1], 
                           c='darkblue', alpha=0.8, s=30, label='OPT (Paired)')

            for i, idx in enumerate(np.where(paired_indices)[0]):
                if i < 50:  
                    axes[1].plot([sar_2d[idx, 0], opt_2d[idx, 0]], 
                               [sar_2d[idx, 1], opt_2d[idx, 1]], 
                               'gray', alpha=0.3, linewidth=0.5)
        
        if np.any(unpaired_indices):
            axes[1].scatter(sar_2d[unpaired_indices, 0], sar_2d[unpaired_indices, 1], 
                           c='lightcoral', alpha=0.6, s=20, label='SAR (Unpaired)')
        
        axes[1].set_title(f'Paired vs Unpaired Features - Epoch {epoch}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'features',
                                f'features_{method}_epoch{epoch:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" 特征可视化已保存: {save_path}")
    
    def visualize_attention_maps(self, model, images, epoch, batch_idx, 
                               image_type="SAR", num_samples=2, layer_idx=-1):

        model.eval()
        device = images.device
        B, C, H, W = images.shape
        num_samples = min(num_samples, B)
        
        with torch.no_grad():

            if hasattr(model, 'sar_encoder'):
                encoder = model.sar_encoder
            else:
                encoder = model

            x = encoder.patch_embed(images[:num_samples])
            if hasattr(encoder, 'pos_embed'):
                x = x + encoder.pos_embed[:, 1:, :]
            if hasattr(encoder, 'cls_token'):
                cls_token = encoder.cls_token + encoder.pos_embed[:, :1, :]
                cls_tokens = cls_token.expand(num_samples, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)

            attentions = []
            for i, blk in enumerate(encoder.blocks):
                if i == len(encoder.blocks) + layer_idx:  

                    B, N, C = x.shape
                    qkv = blk.attn.qkv(blk.norm1(x))
                    qkv = qkv.reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    
                    attn = (q @ k.transpose(-2, -1)) * (C // blk.attn.num_heads) ** -0.5
                    attn = attn.softmax(dim=-1)
                    attentions.append(attn)
                
                x = blk(x)
        
        if not attentions:
            print("️ 未能获取注意力权重")
            return

        attn = attentions[0]  # [B, num_heads, N, N]
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        patch_size = 16
        h = w = H // patch_size
        
        for i in range(num_samples):
            img = images[i].cpu()
            if C == 3:
                img_np = img.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
            else:
                img_np = img.squeeze().numpy()
            
            axes[i, 0].imshow(img_np, cmap='gray' if C == 1 else None)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')

            cls_attn = attn[i, :, 0, 1:].mean(0)  
            cls_attn = cls_attn.cpu().numpy()
            cls_attn = cls_attn.reshape(h, w)

            cls_attn_resized = cv2.resize(cls_attn, (W, H))
            
            im1 = axes[i, 1].imshow(cls_attn_resized, cmap='hot', alpha=0.8)
            axes[i, 1].set_title('CLS Attention')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

            if C == 3:
                overlay = img_np.copy()
                overlay = cv2.addWeighted(overlay.astype(np.float32), 0.7, 
                                        plt.cm.hot(cls_attn_resized)[:,:,:3], 0.3, 0)
            else:
                overlay = plt.cm.hot(cls_attn_resized)[:,:,:3]
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title('Attention Overlay')
            axes[i, 2].axis('off')

            axes[i, 3].hist(cls_attn.flatten(), bins=50, alpha=0.7)
            axes[i, 3].set_title('Attention Distribution')
            axes[i, 3].set_xlabel('Attention Weight')
            axes[i, 3].set_ylabel('Frequency')
        
        plt.suptitle(f'{image_type} Attention Maps - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'attention',
                                f'{image_type}_attention_epoch{epoch:04d}_batch{batch_idx:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" 注意力可视化已保存: {save_path}")
        
        model.train()
    
    def _unpatchify(self, patches, H, W, C, patch_size):

        h = w = H // patch_size
        x = patches.reshape(-1, h, w, patch_size, patch_size, C)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(-1, C, H, W)
        return imgs
    
    def _patchify(self, imgs, patch_size):

        B, C, H, W = imgs.shape
        h = w = H // patch_size
        x = imgs.reshape(B, C, h, patch_size, w, patch_size)
        x = torch.einsum('nchpwq->nhwpqc', x)
        patches = x.reshape(B, h * w, patch_size**2 * C)
        return patches
    
    def _mae_reconstruct_with_mask(self, original, reconstructed_patches, masks, H, W, C, patch_size):

        B, _, _, _ = original.shape
        device = original.device

        original_patches = self._patchify(original, patch_size)  # [B, N, patch_size^2*C]

        reconstructed_full = self._unpatchify(reconstructed_patches, H, W, C, patch_size)  # [B, C, H, W]
        reconstructed_patches_from_full = self._patchify(reconstructed_full, patch_size)  # [B, N, patch_size^2*C]

        masks_expanded = masks.unsqueeze(-1).expand(-1, -1, patch_size**2 * C)  # [B, N, patch_size^2*C]

        final_patches = original_patches * (1 - masks_expanded) + reconstructed_patches_from_full * masks_expanded

        final_imgs = self._unpatchify(final_patches, H, W, C, patch_size)
        
        return final_imgs
    
    def create_comparison_grid(self, sar_images, opt_images, sar_recon, epoch, batch_idx, num_samples=4):

        num_samples = min(num_samples, len(sar_images))
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            sar_img = sar_images[i].cpu()

            sar_img = (sar_img + 1.0) / 2.0
            
            if sar_img.shape[0] == 3:
                sar_np = sar_img.permute(1, 2, 0).numpy()
                sar_np = np.clip(sar_np, 0, 1)
            else:
                sar_np = sar_img.squeeze().numpy()
                sar_np = np.clip(sar_np, 0, 1)
            
            axes[i, 0].imshow(sar_np, cmap='gray')
            axes[i, 0].set_title('SAR Original')
            axes[i, 0].axis('off')

            opt_img = opt_images[i].cpu()

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            opt_img = opt_img * std + mean 
            
            opt_np = opt_img.permute(1, 2, 0).numpy()
            opt_np = np.clip(opt_np, 0, 1)
            
            axes[i, 1].imshow(opt_np)
            axes[i, 1].set_title('Optical')
            axes[i, 1].axis('off')

            if sar_recon is not None:
                recon_img = sar_recon[i].cpu()

                recon_img = (recon_img + 1.0) / 2.0
                
                if recon_img.shape[0] == 3:
                    recon_np = recon_img.permute(1, 2, 0).numpy()
                    recon_np = np.clip(recon_np, 0, 1)
                else:
                    recon_np = recon_img.squeeze().numpy()
                    recon_np = np.clip(recon_np, 0, 1)
                
                axes[i, 2].imshow(recon_np, cmap='gray')
                axes[i, 2].set_title('SAR Reconstructed')
            else:
                axes[i, 2].text(0.5, 0.5, 'No Reconstruction', 
                               ha='center', va='center', transform=axes[i, 2].transAxes)
            axes[i, 2].axis('off')
        
        plt.suptitle(f'SAR-OPT Comparison - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'reconstructions',
                                f'comparison_epoch{epoch:04d}_batch{batch_idx:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" 对比网格已保存: {save_path}")


def create_visualizer(save_dir="./visualizations"):
    return MAEVisualizer(save_dir)
