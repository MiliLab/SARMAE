# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)

        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f" Model output contains NaN/Inf!")
            print(f" Model internal state check:")
            
            nan_params = []
            inf_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params.append(name)
                if torch.isinf(param).any():
                    inf_params.append(name)
            
            if nan_params:
                print(f"   - NaN parameters: {nan_params}")
            if inf_params:
                print(f"   - Inf parameters: {inf_params}")
            
            model.eval()
            with torch.no_grad():
                try:

                    x = samples
                    x = model.patch_embed(x)
                    print(f"   - Patch embed output: min={x.min().item():.6f}, max={x.max().item():.6f}")
                    
                    cls_tokens = model.cls_token.expand(x.shape[0], -1, -1)
                    x = torch.cat((cls_tokens, x), dim=1)
                    x = x + model.pos_embed
                    print(f"   - After pos embed: min={x.min().item():.6f}, max={x.max().item():.6f}")

                    for i, blk in enumerate(model.blocks[:3]):
                        print(f"   - Before Block {i}: min={x.min().item():.6f}, max={x.max().item():.6f}")
                        
                        if torch.isnan(x).any() or torch.isinf(x).any():
                            print(f"   -  Input to Block {i} contains NaN/Inf!")
                            break
                        
                        try:
                            # LayerNorm 1
                            x_norm1 = blk.norm1(x)
                            print(f"   - Block {i} norm1: min={x_norm1.min().item():.6f}, max={x_norm1.max().item():.6f}")
                            
                            # Attention
                            attn_out = blk.attn(x_norm1)
                            print(f"   - Block {i} attention: min={attn_out.min().item():.6f}, max={attn_out.max().item():.6f}")
                            
                            # First residual connection
                            x = x + blk.drop_path(attn_out)
                            print(f"   - Block {i} after first residual: min={x.min().item():.6f}, max={x.max().item():.6f}")
                            
                            # LayerNorm 2
                            x_norm2 = blk.norm2(x)
                            print(f"   - Block {i} norm2: min={x_norm2.min().item():.6f}, max={x_norm2.max().item():.6f}")
                            
                            # MLP
                            mlp_out = blk.mlp(x_norm2)
                            print(f"   - Block {i} mlp: min={mlp_out.min().item():.6f}, max={mlp_out.max().item():.6f}")
                            
                            # Second residual connection
                            x = x + blk.drop_path(mlp_out)
                            print(f"   - Block {i} final output: min={x.min().item():.6f}, max={x.max().item():.6f}")
                            
                        except Exception as e:
                            print(f"   - Error in Block {i} detailed check: {e}")
                        
                        if torch.isnan(x).any() or torch.isinf(x).any():
                            print(f"   -  Block {i} produced NaN/Inf!")
                            break
                except Exception as e:
                    print(f"   - Error during internal check: {e}")
            
            model.train()
            sys.exit(1)
        
        loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f" Loss is {loss_value}, stopping training")
            print(f" Debug info:")
            print(f"   - Sample shape: {samples.shape}")
            print(f"   - Sample min/max: {samples.min().item():.6f}/{samples.max().item():.6f}")
            print(f"   - Sample mean/std: {samples.mean().item():.6f}/{samples.std().item():.6f}")
            print(f"   - Target shape: {targets.shape}")
            print(f"   - Target min/max: {targets.min().item()}/{targets.max().item()}")
            print(f"   - Output shape: {outputs.shape}")
            print(f"   - Output min/max: {outputs.min().item():.6f}/{outputs.max().item():.6f}")
            
            # 检查模型参数
            nan_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    nan_params.append(name)
            if nan_params:
                print(f"   - NaN/Inf parameters: {nan_params}")
            
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    
    # 用于收集所有预测和真实标签，计算每类准确率
    all_predictions = []
    all_targets = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        #  启用混合精度，提升训练性能
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        #  测试阶段的数值稳定性检查
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f" Test Loss is {loss_value}, stopping evaluation")
            print(f" Test Debug info:")
            print(f"   - Images shape: {images.shape}")
            print(f"   - Images min/max: {images.min().item():.6f}/{images.max().item():.6f}")
            print(f"   - Target shape: {target.shape}")
            print(f"   - Target min/max: {target.min().item()}/{target.max().item()}")
            print(f"   - Output shape: {output.shape}")
            print(f"   - Output min/max: {output.min().item():.6f}/{output.max().item():.6f}")
            
            # 检查模型参数
            nan_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    nan_params.append(name)
            if nan_params:
                print(f"   - NaN/Inf parameters: {nan_params}")
            
            # 设置一个默认的loss值，避免程序崩溃
            loss_value = 1e6
            loss = torch.tensor(loss_value, device=loss.device)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        # 获取预测结果
        _, predicted = output.max(1)
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # 计算每类准确率
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    
    # 计算每类准确率
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # 打印每类准确率
    print("\n" + "="*50)
    print("Per-Class Accuracy Report:")
    print("="*50)
    for i, acc in enumerate(per_class_accuracy):
        print(f"Class {i:2d}: {acc*100:.2f}% ({cm.diagonal()[i]:4d}/{cm.sum(axis=1)[i]:4d})")
    
    print(f"\nOverall Accuracy: {np.mean(per_class_accuracy)*100:.2f}%")
    print("="*50)
    
    # 生成详细的分类报告
    class_report = classification_report(all_targets, all_predictions, 
                                       target_names=[f'Class_{i}' for i in range(len(per_class_accuracy))],
                                       digits=4)
    print("\nDetailed Classification Report:")
    print(class_report)

    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['per_class_accuracy'] = per_class_accuracy.tolist()
    results['confusion_matrix'] = cm.tolist()
    results['classification_report'] = class_report
    
    return results


@torch.no_grad()
def evaluate_detailed(data_loader, model, device, class_names=None, save_dir=None):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import json
    
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Detailed Test:'

    # switch to evaluation mode
    model.eval()
    
    # 用于收集所有预测和真实标签
    all_predictions = []
    all_targets = []
    all_probs = []  # 存储预测概率

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # 获取预测概率和预测结果
        probs = torch.softmax(output, dim=1)
        _, predicted = output.max(1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # 获取类别数量
    num_classes = len(np.unique(all_targets))
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    
    # 计算每类准确率
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # 计算每类精确度、召回率、F1分数
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_predictions)
    
    # 打印详细结果
    print("\n" + "="*80)
    print("DETAILED PER-CLASS ACCURACY REPORT")
    print("="*80)
    print(f"{'Class':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-"*80)
    
    for i in range(num_classes):
        print(f"{class_names[i]:<15} {per_class_accuracy[i]*100:>8.2f}% {precision[i]*100:>8.2f}% "
              f"{recall[i]*100:>8.2f}% {f1[i]*100:>8.2f}% {support[i]:>8d}")
    
    # 计算宏平均和加权平均
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    print("-"*80)
    print(f"{'Macro Avg':<15} {macro_precision*100:>8.2f}% {macro_precision*100:>8.2f}% "
          f"{macro_recall*100:>8.2f}% {macro_f1*100:>8.2f}% {np.sum(support):>8d}")
    print(f"{'Weighted Avg':<15} {weighted_precision*100:>8.2f}% {weighted_precision*100:>8.2f}% "
          f"{weighted_recall*100:>8.2f}% {weighted_f1*100:>8.2f}% {np.sum(support):>8d}")
    print("="*80)
    
    # 生成sklearn分类报告
    class_report = classification_report(all_targets, all_predictions, 
                                       target_names=class_names, digits=4)
    print("\nScikit-learn Classification Report:")
    print(class_report)
    
    # 绘制混淆矩阵热图
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制每类准确率柱状图
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(num_classes), per_class_accuracy * 100)
        plt.xlabel('Class')
        plt.ylabel('Accuracy (%)')
        plt.title('Per-Class Accuracy')
        plt.xticks(range(num_classes), class_names, rotation=45, ha='right')
        plt.ylim(0, 100)
        
        # 在柱状图上添加数值
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细结果到JSON文件
        results_dict = {
            'overall_accuracy': float(metric_logger.acc1.global_avg),
            'per_class_accuracy': per_class_accuracy.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'macro_avg': {
                'precision': float(macro_precision),
                'recall': float(macro_recall),
                'f1_score': float(macro_f1)
            },
            'weighted_avg': {
                'precision': float(weighted_precision),
                'recall': float(weighted_recall),
                'f1_score': float(weighted_f1)
            }
        }
        
        with open(os.path.join(save_dir, 'detailed_results.json'), 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        # 保存分类报告
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(class_report)
        
        print(f"\n Detailed results saved to: {save_dir}")
    
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results.update({
        'per_class_accuracy': per_class_accuracy.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1.tolist(),
        'support': support.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'class_names': class_names
    })
    
    return results