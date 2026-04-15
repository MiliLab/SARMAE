import argparse
import datetime
import json
import numpy as np
import os
import time
import signal
import sys
from pathlib import Path

import multiprocessing
try:
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass 

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'
if 'MKL_NUM_THREADS' not in os.environ:
    os.environ['MKL_NUM_THREADS'] = '1'

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

TENSORBOARD_AVAILABLE = False
SummaryWriter = None
import torchvision.transforms as transforms
import torchvision.datasets as datasets

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("️ timm不可用，某些功能可能受限")

from dataset.mixed_dataset import MixedSARDataset
from dataset.transforms import build_sar_transform, build_optical_transform
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.mae_contrastive import SAROpticalPatchAlignment
from models.models_mae import MaskedAutoencoderViT, mae_vit_base_patch16_dec512d8b
from engine_pretrain import train_one_epoch
from util.collate import mixed_sar_collate_fn
from util.monitor import TrainingMonitor, LossWeightScheduler
from util.visualizer import MAEVisualizer
from functools import partial 


def get_args_parser():
    parser = argparse.ArgumentParser('MAE SAR-Optical Patch Alignment pre-training', add_help=False)
    
    # 基本参数
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # 模型参数
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train (mae_vit_base_patch16, mae_vit_large_patch16)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches)')

    # 优化器参数
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--clip_grad', type=float, default=1.0, metavar='NORM',
                        help='gradient clipping max norm (default: 1.0, set to None to disable)')

    # 数据参数
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    # 分布式训练参数
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # 预训练模型路径
    parser.add_argument('--sar_pretrained', default='', type=str,
                        help='SAR pretrained model path')
    parser.add_argument('--optical_pretrained', default='', type=str,
                        help='Optical pretrained model path')
    parser.add_argument('--dinov3_pretrained', default='', type=str,
                        help='DINOv3 pretrained model path (for optical encoder)')
    parser.add_argument('--freeze_optical_completely', action='store_true', default=False,
                        help='Completely freeze optical encoder during training')

    # 损失权重
    parser.add_argument('--mae_loss_weight', default=1.0, type=float,
                        help='Weight for MAE loss')
    parser.add_argument('--alignment_loss_weight', default=1.0, type=float,
                        help='Weight for patch-level alignment loss')
    
    #  消融实验参数
    parser.add_argument('--disable_alignment', action='store_true', default=False,
                        help='Disable patch alignment completely (for ablation study)')
    
    #  损失权重调度参数
    parser.add_argument('--loss_schedule', default='fixed', type=str, choices=['fixed', 'cosine', 'linear', 'step', 'adaptive'],
                        help='Loss weight scheduling strategy')
    
    #  监控和可视化参数
    parser.add_argument('--enable_monitoring', action='store_true', default=True,
                        help='Enable training monitoring and visualization')
    parser.add_argument('--visualize_freq', default=100, type=int,
                        help='Visualization frequency (every N batches)')
    parser.add_argument('--save_vis_samples', default=4, type=int,
                        help='Number of samples to visualize')
    
    #  SAR噪声参数
    parser.add_argument('--enable_sar_noise', action='store_true', default=False,
                        help='Enable SAR image noise for denoising training')
    parser.add_argument('--noise_std', default=0.1, type=float,
                        help='Standard deviation of multiplicative noise for SAR images (fixed mode)')
    parser.add_argument('--noise_schedule', default='fixed', type=str, choices=['fixed', 'linear', 'cosine', 'exponential'],
                        help='Noise strength scheduling strategy')
    parser.add_argument('--initial_noise_std', default=0.15, type=float,
                        help='Initial noise standard deviation (for scheduling)')
    parser.add_argument('--final_noise_std', default=0.05, type=float,
                        help='Final noise standard deviation (for scheduling)')
    
    #  随机噪声参数 
    parser.add_argument('--random_noise', action='store_true', default=True,
                        help='Enable random noise strength for each image (default: True)')
    parser.add_argument('--noise_min', default=0.0, type=float,
                        help='Minimum noise standard deviation (random mode, 0-0.7 range)')
    parser.add_argument('--noise_max', default=0.7, type=float,
                        help='Maximum noise standard deviation (random mode, 0-0.7 range)')
    parser.add_argument('--noise_ratio', default=0.5, type=float,
                        help='Ratio of images to add noise (0.0-1.0), default 0.5 means 50%% images are noisy')
    
    #  消融实验控制 
    parser.add_argument('--ablation_mode', default='full', type=str, 
                        choices=['full', 'mae_only', 'alignment_only', 'no_noise'],
                        help='Ablation study mode: full (MAE+alignment+noise), mae_only (disable alignment), alignment_only (disable noise), no_noise (disable noise only)')
    
    #  原有的消融实验参数（保持兼容性）
    parser.add_argument('--disable_sar_noise_for_ablation', action='store_true', default=False,
                        help='Disable SAR noise for ablation study (overrides enable_sar_noise)')
    
    #  梯度检查点参数 
    parser.add_argument('--use_checkpoint', action='store_true', default=False,
                        help='Enable gradient checkpointing to save GPU memory (slower but uses less memory)')
    parser.add_argument('--checkpoint_encoder', action='store_true', default=False,
                        help='Enable checkpointing only for SAR encoder (recommended)')
    parser.add_argument('--checkpoint_decoder', action='store_true', default=False,
                        help='Enable checkpointing only for SAR decoder')

    # 其他参数
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of data loading workers (0=单进程加载避免多进程同步问题，适合分布式训练)')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)  # 默认禁用pin_memory，避免分布式问题

    # 训练参数
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    return parser

# 全局变量用于清理
_cleanup_resources = {}

def cleanup_training_resources():
    """清理训练资源"""
    global _cleanup_resources
    
    print("\n 执行资源清理...")
    
    # 清理DataLoader
    if 'data_loader' in _cleanup_resources:
        try:
            del _cleanup_resources['data_loader']
            print("   DataLoader已清理")
        except Exception as e:
            print(f"  ️ DataLoader清理失败: {e}")
    
    # 等待workers退出
    time.sleep(1)
    
    # 清理子进程
    try:
        active_children = multiprocessing.active_children()
        if active_children:
            print(f"   清理 {len(active_children)} 个子进程...")
            for child in active_children:
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=2)
                    if child.is_alive():
                        child.kill()
        print("   子进程已清理")
    except Exception as e:
        print(f"  ️ 子进程清理失败: {e}")
    
    # 清理分布式
    if 'distributed' in _cleanup_resources and _cleanup_resources['distributed']:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
            print("   分布式进程组已清理")
        except Exception as e:
            print(f"  ️ 分布式清理失败: {e}")
    
    print(" 资源清理完成\n")

def signal_handler(signum, frame):
    """处理中断信号"""
    print(f"\n️ 收到信号 {signum}，正在清理资源...")
    cleanup_training_resources()
    sys.exit(1)

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 跳过冻结参数
        
        # bias 和 norm 层参数不使用权重衰减
        if len(param.shape) == 1 or name.endswith(".bias") or 'norm' in name or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    
    print(f" 参数分组:")
    print(f"  - 使用权重衰减: {len(decay)} 个参数")
    print(f"  - 不使用权重衰减: {len(no_decay)} 个参数")
    
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]
    
def main(args):
    global _cleanup_resources
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    misc.init_distributed_mode(args)
    
    # 记录分布式状态用于清理
    _cleanup_resources['distributed'] = args.distributed if hasattr(args, 'distributed') else False

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    #  验证噪声参数配置
    if args.enable_sar_noise and args.random_noise:
        if args.noise_schedule != 'fixed':
            print("️  警告：启用随机噪声时，噪声调度器将被忽略")
            print(f"   当前设置：--noise_schedule {args.noise_schedule} (将被忽略)")
            print(f"   随机噪声：σ ∈ [{args.noise_min:.2f}, {args.noise_max:.2f}]")
        
        # 验证随机噪声范围
        if args.noise_min >= args.noise_max:
            raise ValueError(f" noise_min ({args.noise_min}) 必须小于 noise_max ({args.noise_max})")
        
        if args.noise_min < 0.01 or args.noise_max > 1.0:
            print(f"️  警告：噪声范围 [{args.noise_min:.2f}, {args.noise_max:.2f}] 可能不合理")
            if args.noise_max > 1.0:
                print("   建议：noise_max <= 1.0")
            if args.noise_min < 0.01:
                print("   建议：noise_min >= 0.01")
                
        # 验证噪声比例参数
        if args.noise_ratio < 0.0 or args.noise_ratio > 1.0:
            raise ValueError(f"noise_ratio 必须在 [0.0, 1.0] 范围内，当前值: {args.noise_ratio}")
        if args.noise_ratio < 1.0:
            print(f" 混合噪声模式：{args.noise_ratio*100:.1f}% 的图像将被加噪")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    #  确保所有必要参数都有默认值
    if not hasattr(args, 'warmup_epochs'):
        args.warmup_epochs = 1
    if not hasattr(args, 'min_lr'):
        args.min_lr = 1e-6
    if not hasattr(args, 'accum_iter'):
        args.accum_iter = 1
    if not hasattr(args, 'mask_ratio'):
        args.mask_ratio = 0.75
    if not hasattr(args, 'mae_loss_weight'):
        args.mae_loss_weight = 1.0
    if not hasattr(args, 'alignment_loss_weight'):
        args.alignment_loss_weight = 1.0
    
    #  消融实验：如果禁用对齐损失，自动设置权重为0
    if args.disable_alignment:
        args.alignment_loss_weight = 0.0
        print(" 消融实验模式：patch对齐已禁用")
    
    #  处理新的消融实验模式
    if hasattr(args, 'ablation_mode'):
        print(f" 消融实验模式: {args.ablation_mode}")
        
        if args.ablation_mode == 'mae_only':
            # 只保留MAE，禁用patch对齐
            args.alignment_loss_weight = 0.0
            args.disable_alignment = True
            print("   - 禁用patch对齐，只进行MAE训练")
            
        elif args.ablation_mode == 'alignment_only':
            # 禁用噪声，只做patch对齐+MAE
            args.enable_sar_noise = False
            args.disable_sar_noise_for_ablation = True
            print("   - 禁用SAR噪声，进行patch对齐+MAE训练")
            
        elif args.ablation_mode == 'no_noise':
            # 禁用噪声，保留patch对齐
            args.enable_sar_noise = False
            args.disable_sar_noise_for_ablation = True
            print("   - 仅禁用SAR噪声，保留patch对齐")
            
        elif args.ablation_mode == 'full':
            # 完整模式：MAE + patch对齐 + 噪声（如果enable_sar_noise=True）
            print("   - 完整模式：MAE + patch对齐 + 噪声（如果启用）")
    
    #  处理噪声消融实验的覆盖逻辑
    if hasattr(args, 'disable_sar_noise_for_ablation') and args.disable_sar_noise_for_ablation:
        args.enable_sar_noise = False
        print(" 消融实验：SAR噪声已被强制禁用")
    
    #  打印最终的损失权重设置
    print(f" 最终损失权重设置:")
    print(f"   - MAE损失权重: {args.mae_loss_weight}")
    print(f"   - Patch对齐权重: {args.alignment_loss_weight}")
    if args.alignment_loss_weight == 0.0:
        print(f"   ️  Patch对齐已禁用，只进行MAE训练")
    
    #  打印最终的噪声设置
    print(f" 最终噪声设置:")
    print(f"   - SAR噪声启用: {args.enable_sar_noise}")
    if args.enable_sar_noise:
        print(f"   - 随机噪声模式: {args.random_noise}")
        if args.random_noise:
            print(f"   - 噪声范围: [{args.noise_min:.2f}, {args.noise_max:.2f}]")
            print(f"   - 噪声比例: {args.noise_ratio*100:.1f}%")
        else:
            print(f"   - 固定噪声强度: {args.noise_std}")
    else:
        print(f"   ️  SAR噪声已禁用")

    #  数据路径设置
    print("Creating dataset...")
    sar_dir = os.path.join(args.data_path, 'SAR')
    optical_dir = os.path.join(args.data_path, 'OPT') 
    paired_json = os.path.join(args.data_path, 'paired.json')
    unpaired_json = os.path.join(args.data_path, 'unpaired.json')
    
    # 检查路径是否存在
    print(f" 检查数据路径:")
    print(f"  - SAR目录: {sar_dir} {'' if os.path.exists(sar_dir) else ''}")
    print(f"  - 光学目录: {optical_dir} {'' if os.path.exists(optical_dir) else ''}")
    print(f"  - 配对JSON: {paired_json} {'' if os.path.exists(paired_json) else ''}")
    print(f"  - 非配对JSON: {unpaired_json} {'' if os.path.exists(unpaired_json) else ''}")
    
    #  使用自定义transforms
    transform_sar = build_sar_transform(size=args.input_size)
    transform_optical = build_optical_transform(size=args.input_size)
    
    print(f" 使用自定义transforms:")
    print(f"  - SAR transform: 单通道→3通道，ImageNet标准化")
    print(f"  - Optical transform: ImageNet标准化")

    #  计算噪声参数
    if args.enable_sar_noise:
        if args.random_noise:
            #  随机噪声模式：忽略调度器和固定值
            current_noise_std = 0.0  # 在随机模式下此值被忽略
            noise_scheduler = None
            print(f" SAR随机噪声模式:")
            print(f"   - 噪声范围: σ ∈ [{args.noise_min:.2f}, {args.noise_max:.2f}]")
            print(f"   - 每张图像独立随机生成噪声强度")
            
            # 验证噪声范围合理性
            if args.noise_min >= args.noise_max:
                raise ValueError(f"noise_min ({args.noise_min}) 必须小于 noise_max ({args.noise_max})")
            if args.noise_min < 0 or args.noise_max > 1.0:
                print(f"️  噪声范围 [{args.noise_min:.2f}, {args.noise_max:.2f}] 可能过大")
                
        elif args.noise_schedule != 'fixed':
            # 使用调度噪声强度
            from util.sar_noise import create_sar_noise_scheduler
            noise_scheduler = create_sar_noise_scheduler(
                initial_std=args.initial_noise_std,
                final_std=args.final_noise_std,
                total_epochs=args.epochs,
                schedule_type=args.noise_schedule
            )
            current_noise_std = noise_scheduler(0)  # 第0个epoch的噪声强度
            print(f" SAR噪声调度器: {args.noise_schedule}")
            print(f"   - 初始噪声强度: {args.initial_noise_std}")
            print(f"   - 最终噪声强度: {args.final_noise_std}")
            print(f"   - 当前噪声强度: {current_noise_std:.4f}")
        else:
            # 固定噪声强度
            current_noise_std = args.noise_std
            noise_scheduler = None
            print(f" 固定SAR噪声强度: {current_noise_std}")
    else:
        current_noise_std = 0.0
        noise_scheduler = None
        print(" SAR噪声功能已禁用")

    # 创建数据集
    dataset_train = MixedSARDataset(
        sar_dir=sar_dir,
        optical_dir=optical_dir,
        paired_json=paired_json,
        unpaired_json=unpaired_json,
        transform_sar=transform_sar,
        transform_optical=transform_optical,
        paired_ratio=0.7,
        enable_sar_noise=args.enable_sar_noise,
        noise_std=current_noise_std,
        random_noise=args.random_noise,
        noise_range=(args.noise_min, args.noise_max),
        noise_ratio=args.noise_ratio
    )
    
    print(f" 数据集创建成功，样本数量: {len(dataset_train)}")

    # 分布式采样器
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    
    if True:  # args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    #  创建监控和可视化工具
    monitor = None
    visualizer = None
    loss_scheduler = None
    
    if global_rank == 0:
        log_writer = None  
        
        # 创建监控器
        if args.enable_monitoring:
            monitor = TrainingMonitor(
                log_dir=args.log_dir or args.output_dir,
                save_freq=50,
                plot_freq=args.visualize_freq,
                max_history=2000
            )
            
            # 创建可视化器
            vis_dir = os.path.join(args.output_dir, 'visualizations')
            visualizer = MAEVisualizer(save_dir=vis_dir)
            
            print(f" 监控和可视化工具已启用")
        
        # 创建损失权重调度器
        if args.loss_schedule != 'fixed':
            initial_weights = {
                'mae': args.mae_loss_weight,
                'alignment': args.alignment_loss_weight
            }
            loss_scheduler = LossWeightScheduler(
                initial_weights=initial_weights,
                schedule_type=args.loss_schedule,
                total_epochs=args.epochs
            )
            print(f" 损失权重调度器已启用: {args.loss_schedule}")
    else:
        log_writer = None

    #  数据加载器配置 - 优化多进程设置
    world_size = misc.get_world_size()
    if world_size > 1:
        print(f" 检测到分布式训练({world_size}卡)，优化DataLoader设置")
        #  分布式训练优化设置
        args.pin_mem = False  # 分布式训练时禁用pin_memory避免问题
        
        # 限制workers数量避免semaphore泄漏和长时间运行导致的卡死
        max_workers = min(args.num_workers, 4)  # 限制最大workers为4
        if args.num_workers > max_workers:
            print(f"   ️ 将num_workers从{args.num_workers}降低到{max_workers}以避免资源竞争和长时间运行问题")
            args.num_workers = max_workers
            
        print(f"   - num_workers: {args.num_workers} (优化后)")
        print(f"   - batch_size: {args.batch_size} (每GPU)")
        print(f"   - 总batch_size: {args.batch_size * world_size}")
        print(f"   - 禁用pin_memory: {args.pin_mem}")
    else:
        print(f" 单卡训练，使用用户指定设置:")
        print(f"   - num_workers: {args.num_workers}")
        print(f"   - batch_size: {args.batch_size}")
        print(f"   - pin_memory: {args.pin_mem}")
    
    #  使用更保守的DataLoader设置
    try:
        # 根据num_workers选择不同的参数
        if args.num_workers > 0:
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
                collate_fn=mixed_sar_collate_fn,
                persistent_workers=False,  
                prefetch_factor=None,  
                timeout=120,  
                multiprocessing_context='spawn',  
            )
        else:
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=0,
                pin_memory=args.pin_mem,
                drop_last=True,
                collate_fn=mixed_sar_collate_fn
            )
        print(f" DataLoader创建成功，使用 {args.num_workers} 个workers ({world_size}卡训练)")
        print(f"   - persistent_workers: False (避免semaphore泄漏)")
        print(f"   - timeout: 120s")
    except Exception as e:
        print(f"️ 多进程DataLoader创建失败: {e}")
        print(" 回退到单进程模式...")
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            collate_fn=mixed_sar_collate_fn
        )
    
    #  将data_loader存入全局资源用于清理
    _cleanup_resources['data_loader'] = data_loader_train

    print("Creating model...")

    use_checkpoint = args.use_checkpoint
    checkpoint_encoder = args.checkpoint_encoder
    checkpoint_decoder = args.checkpoint_decoder
    
    if checkpoint_encoder or checkpoint_decoder:
        use_checkpoint = True
    
    print(f" Checkpoint设置:")
    print(f"   - 启用checkpoint: {use_checkpoint}")
    print(f"   - Encoder checkpoint: {checkpoint_encoder}")
    print(f"   - Decoder checkpoint: {checkpoint_decoder}")
    if use_checkpoint:
        print(f"   ️  Checkpoint会降低训练速度但节省显存")
    
    print(" 创建基础SAR MAE模型...")

    if args.model == 'mae_vit_large_patch16':
        print(" 使用ViT-Large配置")
        sar_mae_model = MaskedAutoencoderViT(
            img_size=args.input_size,     
            patch_size=16,
            in_chans=3,                   
            embed_dim=1024,              # Large: 1024
            depth=24,                    # Large: 24
            num_heads=16,                # Large: 16
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            use_checkpoint=use_checkpoint  
        )
    else:  # mae_vit_base_patch16
        print(" 使用ViT-Base配置")
        sar_mae_model = MaskedAutoencoderViT(
            img_size=args.input_size,     
            patch_size=16,
            in_chans=3,                   
            embed_dim=768,               # Base: 768
            depth=12,                    # Base: 12
            num_heads=12,                # Base: 12
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            use_checkpoint=use_checkpoint  
        )
    print(f" SAR MAE模型创建成功")
    
    print(" 创建SAR-Optical patch对齐模型...")

    dinov3_path = args.dinov3_pretrained if args.dinov3_pretrained else None
    optical_path = args.optical_pretrained if args.optical_pretrained else None

    if dinov3_path:
        print(f" 将使用DINOv3权重初始化光学编码器: {dinov3_path}")
        if args.freeze_optical_completely:
            print("️ 光学编码器将被完全冻结")
        else:
            print(f" 光学编码器将部分冻结（前{args.freeze_optical_layers if hasattr(args, 'freeze_optical_layers') else 8}层）")
    
    model = SAROpticalPatchAlignment(
        mae_model=sar_mae_model,
        optical_encoder='vit_base_patch16_224',
        freeze_optical_layers=8,
        patch_size=16,
        mae_loss_weight=args.mae_loss_weight,  
        alignment_loss_weight=args.alignment_loss_weight,  
        dinov3_pretrained_path=dinov3_path,
        freeze_optical_completely=args.freeze_optical_completely,
        use_checkpoint=use_checkpoint  
    )

    if args.sar_pretrained:
        print(f" 加载SAR预训练权重: {args.sar_pretrained}")
        model.load_sar_pretrained(args.sar_pretrained)
    else:
        print("️ 未指定SAR预训练权重")

    if optical_path and not dinov3_path:
        print(f" 加载Optical预训练权重: {optical_path}")
        model.load_optical_pretrained(optical_path)
    elif dinov3_path:
        print(" 已通过DINOv3权重初始化光学编码器")
    else:
        print("️ 未指定光学编码器预训练权重")

    model.to(device)
    model_without_ddp = model
    print(" 模型创建完成")
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # 分布式训练
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=False  
        )
        model_without_ddp = model.module
    
    # 优化器
    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)  
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(f" 优化器创建后的初始学习率: {optimizer.param_groups[0]['lr']}")
    
    # loss scaler
    loss_scaler = NativeScaler()

    # 加载checkpoint（如果有resume）
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    try:
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            if args.enable_sar_noise and noise_scheduler is not None and not args.random_noise:
                new_noise_std = noise_scheduler(epoch)
                dataset_train.noise_generator.set_noise_std(new_noise_std)
                if epoch % 10 == 0 or epoch < 5:  # 前5个epoch和每10个epoch打印
                    print(f" Epoch {epoch}: 更新噪声强度为 {new_noise_std:.4f}")
            elif args.enable_sar_noise and args.random_noise and epoch == 0:
                print(f" 随机噪声模式：每张图像独立生成 σ ∈ [{args.noise_min:.2f}, {args.noise_max:.2f}]")

            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                log_writer=None, 
                args=args,
                monitor=monitor,
                loss_scheduler=loss_scheduler
            )

            if dist.is_initialized():
                torch.cuda.synchronize()  # 同步CUDA操作
                if global_rank == 0:
                    print(f" Epoch {epoch} 训练完成，CUDA已同步")

            if monitor is not None and global_rank == 0:
                monitor.log_epoch(epoch, train_stats)

                if epoch % 10 == 0:
                    monitor.log_model_stats(model_without_ddp, epoch)

            if visualizer is not None and epoch % 5 == 0 and global_rank == 0:
                print(f" 生成可视化 (Epoch {epoch})...")

                model_without_ddp.eval()
                with torch.no_grad():
                    for batch_idx, batch_data in enumerate(data_loader_train):
                        if batch_idx > 0:  # 只用第一个batch
                            break
                            
                        sar_batch = batch_data['sar'].to(device)
                        sar_target_batch = batch_data['sar_target'].to(device)  #  获取干净SAR目标
                        is_paired_flags = batch_data['is_paired']
                        optical_batch = None
                        if 'optical' in batch_data:
                            optical_batch = batch_data['optical'].to(device)
                        
                        vis_samples = min(4, sar_batch.size(0))
                        vis_sar = sar_batch[:vis_samples]
                        vis_sar_target = sar_target_batch[:vis_samples]
                        vis_is_paired = is_paired_flags[:vis_samples]
                        vis_optical = optical_batch[:vis_samples] if optical_batch is not None else None

                        if hasattr(model_without_ddp, 'sar_encoder'):
                            mae_model = model_without_ddp.sar_encoder
                        else:
                            mae_model = model_without_ddp

                        latent, mask, ids_restore = mae_model.forward_encoder(vis_sar, args.mask_ratio)
                        pred = mae_model.forward_decoder(latent, ids_restore)

                        visualizer.visualize_masks(
                            vis_sar, mask, ids_restore, epoch, 0, 
                            image_type="SAR", num_samples=vis_samples
                        )

                        if vis_optical is not None:
                            visualizer.visualize_masks(
                                vis_optical, mask, ids_restore, epoch, 0, 
                                image_type="OPT", num_samples=vis_samples
                            )

                        visualizer.visualize_reconstruction(
                            vis_sar_target, 
                            pred, mask, epoch, 0,
                            image_type="SAR", num_samples=vis_samples
                        )

                        if vis_optical is not None:

                            paired_mask = torch.tensor(vis_is_paired, dtype=torch.bool)
                            
                            if torch.any(paired_mask):

                                paired_indices = torch.where(paired_mask)[0]
                                if len(paired_indices) > 0:
                                    paired_sar = vis_sar[paired_indices]
                                    paired_optical = vis_optical[paired_indices]
                                    paired_pred = pred[paired_indices]
                                    
                                    visualizer.create_comparison_grid(
                                        paired_sar, paired_optical, 
                                        visualizer._unpatchify(paired_pred, paired_sar.shape[2], paired_sar.shape[3], 3, 16),
                                        epoch, 0, num_samples=len(paired_indices)
                                    )
                                    print(f" 配对样本可视化完成: {len(paired_indices)} 个样本")

                            unpaired_count = torch.sum(~paired_mask).item()
                            if unpaired_count > 0:
                                print(f" 无配对样本: {unpaired_count} 个（仅参与MAE重建，不参与patch对齐）")
                        
                        print(f" Epoch {epoch} 统一可视化完成")
                        break
                
                model_without_ddp.train()

            if dist.is_initialized():
                torch.cuda.synchronize()
                if global_rank != 0:
                    time.sleep(5)  
                elif global_rank == 0:
                    print(f" 准备保存checkpoint (Epoch {epoch})")

            if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

            if args.output_dir and global_rank == 0:
                try:
                    temp_checkpoint_path = os.path.join(args.output_dir, 'checkpoint-last.pth.tmp')
                    last_checkpoint_path = os.path.join(args.output_dir, 'checkpoint-last.pth')
                    
                    to_save = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }

                    torch.save(to_save, temp_checkpoint_path)

                    import shutil
                    shutil.move(temp_checkpoint_path, last_checkpoint_path)
                    
                    print(f" 已保存最新checkpoint: checkpoint-last.pth (Epoch {epoch})")
                except Exception as e:
                    print(f"️ 保存last checkpoint失败: {e}")

            if dist.is_initialized():
                if global_rank != 0:
                    time.sleep(3) 
                elif global_rank == 0:
                    print(f" Checkpoint保存完成 (Epoch {epoch})")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,}

            if args.output_dir and misc.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
    
    except KeyboardInterrupt:
        print("\n️ 训练被用户中断")
    except Exception as e:
        print(f"\n 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n 开始清理资源...")
        cleanup_training_resources()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if monitor is not None:
        try:
            monitor.close()
            print(" 训练监控器已关闭")
        except Exception as e:
            print(f"️ 监控器清理失败: {e}")
    
    print(" 训练完成，所有资源已清理")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
