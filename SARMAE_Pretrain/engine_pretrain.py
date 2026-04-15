import math
import sys
from typing import Iterable
import numpy as np

import torch
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched
from util.monitor import TrainingMonitor, LossWeightScheduler

def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler, 
                   log_writer=None, args=None, monitor=None, loss_scheduler=None):
    
    if epoch == 0:
        print(f" train_one_epoch 接收到的 args:")
        print(f"  - args.lr: {getattr(args, 'lr', '未设置')}")
        print(f"  - args.min_lr: {getattr(args, 'min_lr', '未设置')}")
        print(f"  - args.warmup_epochs: {getattr(args, 'warmup_epochs', '未设置')}")
        print(f"  - epoch: {epoch}")
    
    model.train(True)
    
    # 调用学习率调度器
    lr_sched.adjust_learning_rate(optimizer, epoch, args)
    current_lr = optimizer.param_groups[0]['lr']
    if epoch % 10 == 0:
        print(f" Epoch {epoch} 学习率设置为: {current_lr:.6f}")
    
    # 更新损失权重调度器
    current_loss_weights = {}
    if loss_scheduler is not None:
        current_loss_weights = loss_scheduler.step(epoch)
        loss_scheduler.print_weights()
    else:
        current_loss_weights = {
            'mae': getattr(args, 'mae_loss_weight', 1.0),
            'alignment': getattr(args, 'alignment_loss_weight', 0.5)
        }
        if epoch == 0:  # 只在第一个epoch打印
            print(f" 使用静态损失权重: MAE={current_loss_weights['mae']}, 对齐损失={current_loss_weights['alignment']}")
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    for data_iter_step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # 调整学习率
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        #  处理批次数据
        sar_batch = batch_data['sar'].to(device, non_blocking=True)
        sar_target_batch = batch_data['sar_target'].to(device, non_blocking=True)  # 重建目标
        is_paired_flags = batch_data['is_paired']
        
        optical_batch = None
        if 'optical' in batch_data:
            optical_batch = batch_data['optical'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            total_loss, loss_dict, alignment_loss = model(
                sar_batch,           # 输入SAR图像（可能加噪）
                optical_batch,       # 光学图像
                is_paired_flags,     # 传递配对标志
                sar_target_batch,    # SAR重建目标（干净图像）
                mask_ratio=args.mask_ratio,
                loss_weights=current_loss_weights  # 传递损失权重
            )

        # 检查损失是否有效
        if not math.isfinite(total_loss.item()):
            print(f"️ Loss is NaN or Inf: {total_loss.item()}")
            print(f"   MAE loss: {loss_dict.get('mae_loss', 'N/A')}")
            print(f"   Alignment loss: {loss_dict.get('alignment_loss', 'N/A')}")
            print(f"   Total loss breakdown: {loss_dict.get('total_loss', 'N/A')}")
            optimizer.zero_grad()
            continue

        total_loss /= accum_iter

        # 反向传播 - 添加梯度裁剪
        clip_grad = getattr(args, 'clip_grad', None)
        loss_scaler(total_loss, optimizer, 
                    clip_grad=clip_grad,
                    parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        #  记录损失 
        total_loss_value = total_loss.item()
        metric_logger.update(loss=total_loss_value)
        
        # 记录各项子损失
        if 'mae_loss' in loss_dict:
            metric_logger.update(mae_loss=loss_dict['mae_loss'])
        if 'alignment_loss' in loss_dict:
            metric_logger.update(alignment_loss=loss_dict['alignment_loss'])

        # 记录学习率
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        #  监控记录 - 批次级别（减少频率）
        if monitor is not None and data_iter_step % 50 == 0:  # 每50个batch记录一次
            # 记录梯度信息
            if (data_iter_step + 1) % accum_iter == 0:
                grad_norm = monitor.log_gradients(model, epoch, data_iter_step)
            
            # 记录批次信息
            batch_losses = {
                'loss': total_loss_value,
                **loss_dict
            }
            monitor.log_batch(batch_losses, lr, epoch, data_iter_step, len(data_loader))

    if misc.get_rank() == 0 and epoch % 10 == 0:  
        print(" Local stats (rank 0):", metric_logger)
    return {k: meter.avg for k, meter in metric_logger.meters.items()}