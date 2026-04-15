import os
import json
import time
import torch
import numpy as np
from collections import defaultdict, deque

TENSORBOARD_AVAILABLE = False

class SummaryWriter:
    def __init__(self, *args, **kwargs):
        pass
    def add_scalar(self, *args, **kwargs):
        pass
    def add_scalars(self, *args, **kwargs):
        pass
    def close(self):
        pass

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True

    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        pass
    
    sns.set_palette("husl")
except ImportError:
    PLOTTING_AVAILABLE = False

    class DummyPlt:
        def figure(self, *args, **kwargs): return self
        def subplots(self, *args, **kwargs): return self, self
        def plot(self, *args, **kwargs): pass
        def scatter(self, *args, **kwargs): pass
        def xlabel(self, *args, **kwargs): pass
        def ylabel(self, *args, **kwargs): pass
        def title(self, *args, **kwargs): pass
        def legend(self, *args, **kwargs): pass
        def grid(self, *args, **kwargs): pass
        def savefig(self, *args, **kwargs): pass
        def close(self, *args, **kwargs): pass
        def show(self, *args, **kwargs): pass
    
    plt = DummyPlt()
    sns = DummyPlt()


class TrainingMonitor:

    def __init__(self, log_dir, save_freq=50, plot_freq=100, max_history=1000):

        self.log_dir = log_dir
        self.save_freq = save_freq
        self.plot_freq = plot_freq
        self.max_history = max_history

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'logs'), exist_ok=True)

        self.writer = SummaryWriter(log_dir) 

        self.history = defaultdict(lambda: deque(maxlen=max_history))
        self.epoch_history = defaultdict(list)
        self.batch_count = 0
        self.epoch_count = 0

        self.start_time = time.time()
        self.epoch_start_time = time.time()
        
        print(f" 训练监控器初始化完成")
        print(f"   - 日志目录: {log_dir}")
        print(f"   - 保存频率: 每{save_freq}个batch")
        print(f"   - 绘图频率: 每{plot_freq}个batch")
    
    def log_batch(self, losses, lr, epoch, batch_idx, total_batches):

        self.batch_count += 1
        current_time = time.time()

        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.history[key].append(value)
            self.writer.add_scalar(f'batch/{key}', value, self.batch_count)

        self.history['lr'].append(lr)
        self.writer.add_scalar('batch/lr', lr, self.batch_count)

        elapsed = current_time - self.epoch_start_time
        eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
    
    def log_epoch(self, epoch, train_stats, val_stats=None):

        self.epoch_count = epoch
        self.epoch_start_time = time.time()

        for key, value in train_stats.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.epoch_history[f'train_{key}'].append(value)
            self.writer.add_scalar(f'epoch/train_{key}', value, epoch)

        if val_stats:
            for key, value in val_stats.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.epoch_history[f'val_{key}'].append(value)
                self.writer.add_scalar(f'epoch/val_{key}', value, epoch)

        self._save_epoch_stats(epoch, train_stats, val_stats)
        
        print(f"\n Epoch {epoch} 完成:")
        print(f"   - 训练损失: {train_stats.get('loss', 'N/A'):.4f}")
        if val_stats:
            print(f"   - 验证损失: {val_stats.get('loss', 'N/A'):.4f}")
    
    def log_gradients(self, model, epoch, batch_idx):

        total_norm = 0
        param_count = 0
        grad_dict = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

                module_name = name.split('.')[0]
                if module_name not in grad_dict:
                    grad_dict[module_name] = []
                grad_dict[module_name].append(param_norm.item())
        
        total_norm = total_norm ** (1. / 2)

        self.writer.add_scalar('gradients/total_norm', total_norm, self.batch_count)
        
        for module_name, norms in grad_dict.items():
            avg_norm = np.mean(norms)
            self.writer.add_scalar(f'gradients/{module_name}', avg_norm, self.batch_count)

        self.history['grad_norm'].append(total_norm)
        
        return total_norm
    
    def log_model_stats(self, model, epoch):

        total_params = 0
        trainable_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"\n 模型统计 (Epoch {epoch}):")
        print(f"   - 总参数量: {total_params:,}")
        print(f"   - 可训练参数: {trainable_params:,}")
        print(f"   - 可训练比例: {trainable_params/total_params:.1%}")

        self.writer.add_scalar('model/total_params', total_params, epoch)
        self.writer.add_scalar('model/trainable_params', trainable_params, epoch)
    
    def _print_batch_info(self, epoch, batch_idx, total_batches, losses, lr, eta):

        progress = (batch_idx + 1) / total_batches * 100
        
        print(f"\n Epoch {epoch} [{batch_idx+1}/{total_batches}] ({progress:.1f}%)")
        print(f"   - 学习率: {lr:.6f}")
        print(f"   - ETA: {eta/60:.1f}分钟")

        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            print(f"   - {key}: {value:.4f}")
    
    def _plot_training_curves(self):

        if len(self.history['loss']) < 10: 
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress (Batch {self.batch_count})', fontsize=16)

        ax1 = axes[0, 0]
        if 'loss' in self.history:
            ax1.plot(list(self.history['loss']), label='Total Loss', alpha=0.7)
        if 'mae_loss' in self.history:
            ax1.plot(list(self.history['mae_loss']), label='MAE Loss', alpha=0.7)
        if 'contrastive_loss' in self.history:
            ax1.plot(list(self.history['contrastive_loss']), label='Contrastive Loss', alpha=0.7)
        
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        if 'lr' in self.history:
            ax2.plot(list(self.history['lr']), color='orange', alpha=0.7)
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('LR')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        if 'grad_norm' in self.history:
            ax3.plot(list(self.history['grad_norm']), color='red', alpha=0.7)
        ax3.set_title('Gradient Norm')
        ax3.set_xlabel('Batch')
        ax3.set_ylabel('Norm')
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        if 'mae_loss' in self.history and 'contrastive_loss' in self.history:
            mae_losses = list(self.history['mae_loss'])
            cont_losses = list(self.history['contrastive_loss'])
            if len(mae_losses) > 0 and len(cont_losses) > 0:
                ratios = [c/(m+1e-8) for m, c in zip(mae_losses, cont_losses)]
                ax4.plot(ratios, color='purple', alpha=0.7)
        ax4.set_title('Contrastive/MAE Ratio')
        ax4.set_xlabel('Batch')
        ax4.set_ylabel('Ratio')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()

        plot_path = os.path.join(self.log_dir, 'plots', f'training_curves_batch_{self.batch_count}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_epoch_stats(self, epoch, train_stats, val_stats):

        stats = {
            'epoch': epoch,
            'train': train_stats,
            'val': val_stats if val_stats else {},
            'timestamp': time.time()
        }
        
        stats_path = os.path.join(self.log_dir, 'logs', f'epoch_{epoch:04d}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def save_final_plots(self):

        if not self.epoch_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Final Training Summary', fontsize=16)
        
        epochs = range(len(list(self.epoch_history.values())[0]))

        ax1 = axes[0, 0]
        for key, values in self.epoch_history.items():
            if 'loss' in key:
                ax1.plot(epochs, values, label=key, marker='o', markersize=3)
        ax1.set_title('Epoch Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        final_plot_path = os.path.join(self.log_dir, 'final_training_summary.png')
        plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" 最终训练总结图已保存: {final_plot_path}")
    
    def close(self):

        self.save_final_plots()
        self.writer.close()
        
        total_time = time.time() - self.start_time
        print(f"\n 训练监控完成，总用时: {total_time/3600:.2f}小时")


class LossWeightScheduler:

    def __init__(self, initial_weights, schedule_type='cosine', total_epochs=100):

        self.initial_weights = initial_weights.copy()
        self.current_weights = initial_weights.copy()
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs
        self.epoch = 0

        self.loss_history = defaultdict(list)
        self.patience = 10
        self.adjustment_factor = 0.9
        
        print(f" 损失权重调度器初始化:")
        print(f"   - 调度类型: {schedule_type}")
        print(f"   - 初始权重: {initial_weights}")
    
    def step(self, epoch, loss_dict=None):

        self.epoch = epoch
        
        if self.schedule_type == 'cosine':
            self._cosine_schedule()
        elif self.schedule_type == 'linear':
            self._linear_schedule()
        elif self.schedule_type == 'step':
            self._step_schedule()
        elif self.schedule_type == 'adaptive':
            self._adaptive_schedule(loss_dict)

        return self.current_weights
    
    def _cosine_schedule(self):

        progress = self.epoch / self.total_epochs
        
        for key in self.current_weights:
            if key == 'contrastive':

                min_weight = 0.1
                max_weight = self.initial_weights[key]
                self.current_weights[key] = min_weight + (max_weight - min_weight) * (
                    0.5 * (1 + np.cos(np.pi * (1 - progress)))
                )
    
    def _linear_schedule(self):

        progress = self.epoch / self.total_epochs
        
        for key in self.current_weights:
            if key == 'contrastive':

                self.current_weights[key] = self.initial_weights[key] * (0.1 + 0.9 * progress)
    
    def _step_schedule(self):

        if self.epoch < self.total_epochs * 0.3:

            self.current_weights['contrastive'] = self.initial_weights['contrastive'] * 0.1
        elif self.epoch < self.total_epochs * 0.7:

            self.current_weights['contrastive'] = self.initial_weights['contrastive'] * 0.5
        else:

            self.current_weights['contrastive'] = self.initial_weights['contrastive'] * 1.0
    
    def _adaptive_schedule(self, loss_dict):

        if loss_dict is None:
            return

        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.loss_history[key].append(value)

        if len(self.loss_history.get('loss', [])) % self.patience == 0:
            self._adjust_weights_adaptively()
    
    def _adjust_weights_adaptively(self):

        if len(self.loss_history['contrastive']) >= self.patience * 2:
            recent_losses = self.loss_history['contrastive'][-self.patience:]
            older_losses = self.loss_history['contrastive'][-self.patience*2:-self.patience]
            
            recent_avg = np.mean(recent_losses)
            older_avg = np.mean(older_losses)

            if recent_avg / older_avg > 0.95:  
                self.current_weights['contrastive'] *= (1 / self.adjustment_factor)
                print(f" 自适应增加对比学习权重: {self.current_weights['contrastive']:.4f}")
    
    def get_current_weights(self):

        return self.current_weights.copy()
    
    def print_weights(self):

        print(f" 当前损失权重 (Epoch {self.epoch}):")
        for key, weight in self.current_weights.items():
            print(f"   - {key}: {weight:.4f}")
