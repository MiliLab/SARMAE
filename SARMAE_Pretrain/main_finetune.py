# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modifications Copyright (c) 2026 Danxu Liu.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit

from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    
    parser.add_argument('--freeze_blocks', type=int, default=0,
                        help='Number of blocks to freeze from the beginning. 0 means no freezing')
    parser.add_argument('--freeze_patch_embed', action='store_true', default=False,
                        help='Freeze patch embedding layer')

    parser.add_argument('--dataset', default='imagenet', type=str,
                        help='dataset name')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    
    parser.add_argument('--split', default=None, type=int,
                        help='train-test split ratio (for custom datasets)')
    parser.add_argument('--tag', default=None, type=int,
                        help='dataset tag or index')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--exp_num', default=1, type=int,
                        help='Number of experiment runs with different random seeds')
    parser.add_argument('--eval_freq', default=1, type=int,
                        help='Frequency of evaluation (epochs)')
    parser.add_argument('--postfix', default='default', type=str,
                        help='Postfix for output directory name')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    if args.output_dir == './output_dir':
        base_name = f"{args.model}_{args.dataset}"
        if args.split:
            base_name += f"_split{args.split}"
        base_name += f"_size{args.input_size}"
        if args.postfix != 'default':
            base_name += f"_{args.postfix}"
        
        if args.finetune:
            pretrain_dir = os.path.dirname(args.finetune)
            args.output_dir = os.path.join(pretrain_dir, f"{base_name}_finetune")
        else:
            args.output_dir = f"./output/{base_name}"
    
    if misc.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.output_dir, "log.txt"), mode="w", encoding="utf-8") as f:
            f.write(f"Starting experiments: {args.exp_num} runs\n")
            f.write(f"Model: {args.model}, Dataset: {args.dataset}\n")
            f.write(f"Epochs: {args.epochs}, Batch size: {args.batch_size}\n")
            f.write(f"Input size: {args.input_size}\n")
            f.write("="*80 + "\n\n")
    
    if args.distributed:
        torch.distributed.barrier()

    exp_record = np.zeros([3, args.exp_num + 2])
    
    for exp_idx in range(args.exp_num):
        
        if misc.is_main_process():
            print('\n' + '='*80)
            print(f'Experiment {exp_idx+1}/{args.exp_num}')
            print('='*80 + '\n')
            
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(f'\n{"="*80}\n')
                f.write(f'Experiment {exp_idx+1}/{args.exp_num}\n')
                f.write(f'{"="*80}\n\n')

        seed = exp_idx + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        cudnn.benchmark = True

        dataset_train = build_dataset(is_train=True, args=args)
        dataset_val = build_dataset(is_train=False, args=args)

        if misc.is_main_process():
            print(f"Train dataset: {len(dataset_train)} samples")
            print(f"Val dataset: {len(dataset_val)} samples")

        global_rank = misc.get_rank()
        if args.distributed:
            num_tasks = misc.get_world_size()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            
            if args.dist_eval:
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if global_rank == 0 and args.log_dir is not None and not args.eval:
            log_dir_exp = os.path.join(args.log_dir, f'exp{exp_idx}')
            os.makedirs(log_dir_exp, exist_ok=True)
            log_writer = SummaryWriter(log_dir=log_dir_exp)
        else:
            log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)

        print(f"Creating model: {args.model}")
        
        if args.model.startswith('dinov3'):
            if args.model == 'dinov3_vit_small_patch14':
                model = models_vit.dinov3_vit_small_patch14(
                    num_classes=args.nb_classes,
                    img_size=args.input_size,
                    drop_path_rate=args.drop_path,
                    global_pool=args.global_pool,
                )
            elif args.model == 'dinov3_vit_base_patch14':
                model = models_vit.dinov3_vit_base_patch14(
                    num_classes=args.nb_classes,
                    img_size=args.input_size,
                    drop_path_rate=args.drop_path,
                    global_pool=args.global_pool,
                )
            elif args.model == 'dinov3_vit_base_patch16':
                model = models_vit.dinov3_vit_base_patch16(
                    num_classes=args.nb_classes,
                    img_size=args.input_size,
                    drop_path_rate=args.drop_path,
                    global_pool=args.global_pool,
                )
            elif args.model == 'dinov3_vit_large_patch14':
                model = models_vit.dinov3_vit_large_patch14(
                    num_classes=args.nb_classes,
                    img_size=args.input_size,
                    drop_path_rate=args.drop_path,
                    global_pool=args.global_pool,
                )
            else:
                raise ValueError(f"Unknown DINOv3 model: {args.model}")
        
        elif args.model == 'vit_large_patch16':
            model = models_vit.vit_large_patch16(
                num_classes=args.nb_classes,
                drop_path_rate=args.drop_path,
                global_pool=args.global_pool,
            )
        elif args.model == 'vit_base_patch16':
            model = models_vit.vit_base_patch16(
                num_classes=args.nb_classes,
                drop_path_rate=args.drop_path,
                global_pool=args.global_pool,
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")

        if args.finetune and not args.eval:
            print(f"Loading pretrained model from: {args.finetune}")
            
            if 'dinov' in args.finetune.lower() or args.model.startswith('dinov3'):
                from util.misc import load_dinov3_pretrain, reinit_head
                
                load_dinov3_pretrain(model, args.finetune)
                reinit_head(model)
                
                print("DINOv3 weights loaded successfully")
            else:
                checkpoint = torch.load(args.finetune, map_location='cpu')
                checkpoint_model = checkpoint['model']
                
                interpolate_pos_embed(model, checkpoint_model)
                
                msg = model.load_state_dict(checkpoint_model, strict=False)
                print(msg)
                
                if msg.missing_keys:
                    print(f"Missing keys when loading finetune checkpoint: {msg.missing_keys}")
                if msg.unexpected_keys:
                    print(f"Unexpected keys when loading finetune checkpoint: {msg.unexpected_keys}")

                if hasattr(model, "head") and hasattr(model.head, "weight"):
                    trunc_normal_(model.head.weight, std=2e-5)

        if args.freeze_blocks > 0:
            print(f"Freezing first {args.freeze_blocks} blocks")
            
            if args.freeze_patch_embed:
                for param in model.patch_embed.parameters():
                    param.requires_grad = False
                print("   Patch embedding frozen")
            
            if hasattr(model, 'blocks'):
                for i in range(min(args.freeze_blocks, len(model.blocks))):
                    for param in model.blocks[i].parameters():
                        param.requires_grad = False
                print(f"   Blocks 0-{args.freeze_blocks-1} frozen")
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

        model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Trainable params (M): {n_parameters / 1e6:.2f}")

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        
        if args.lr is None:
            args.lr = args.blr * eff_batch_size / 256

        print(f"Base lr: {args.blr:.2e}")
        print(f"Actual lr: {args.lr:.2e}")
        print(f"Effective batch size: {eff_batch_size}")

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        param_groups = lrd.param_groups_lrd(
            model_without_ddp, 
            args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        loss_scaler = NativeScaler()

        if mixup_fn is not None:
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        print(f"Criterion: {criterion}")

        if args.eval:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy: {test_stats['acc1']:.1f}%")
            exit(0)

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                log_writer=log_writer,
                args=args
            )
            
            if epoch % args.eval_freq == 0 or epoch + 1 == args.epochs:
                test_stats = evaluate(data_loader_val, model, device)
                
                if test_stats["acc1"] > max_accuracy:
                    max_accuracy = test_stats["acc1"]
                
                print(f"Epoch {epoch}: Acc={test_stats['acc1']:.2f}% | Max={max_accuracy:.2f}%")

                if misc.is_main_process() and test_stats["acc1"] == max_accuracy:
                    checkpoint_path = os.path.join(args.output_dir, f'best_ckpt_exp{exp_idx}.pth')
                    checkpoint = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        'max_accuracy': max_accuracy,
                    }
                    print(f"Saving checkpoint to {checkpoint_path}")
                    torch.save(checkpoint, checkpoint_path)

                if log_writer is not None:
                    log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                    log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters
                }

                if misc.is_main_process():
                    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        if misc.is_main_process():
            print(f'Training time: {total_time_str}')

        if exp_idx < exp_record.shape[1] - 2:
            exp_record[0, exp_idx] = test_stats['acc1']
            exp_record[1, exp_idx] = max_accuracy
            exp_record[2, exp_idx] = int(total_time)
            
            if misc.is_main_process():
                print(f"Exp {exp_idx+1}: Last={test_stats['acc1']:.2f}%, Max={max_accuracy:.2f}%, Time={total_time_str}")
        
        del model, optimizer, loss_scaler, data_loader_train, data_loader_val
        if args.distributed:
            torch.distributed.barrier()
            torch.cuda.empty_cache()
        
        if misc.is_main_process():
            print(f"Experiment {exp_idx+1}/{args.exp_num} completed\n")
    
    if misc.is_main_process():
        valid_exps = min(args.exp_num, exp_record.shape[1] - 2)
        
        exp_record[0, -2] = np.mean(exp_record[0, :valid_exps])
        exp_record[0, -1] = np.std(exp_record[0, :valid_exps])
        exp_record[1, -2] = np.mean(exp_record[1, :valid_exps])
        exp_record[1, -1] = np.std(exp_record[1, :valid_exps])
        exp_record[2, -2] = np.mean(exp_record[2, :valid_exps])
        exp_record[2, -1] = np.std(exp_record[2, :valid_exps])

        np.save(os.path.join(args.output_dir, 'exp_record.npy'), exp_record)
        
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write('\n' + '='*80 + '\n')
            f.write('FINAL RESULTS\n')
            f.write('='*80 + '\n')
            f.write(f'Last Acc1: {exp_record[0,-2]:.2f} +/- {exp_record[0,-1]:.2f}%\n')
            f.write(f'Max Acc1: {exp_record[1,-2]:.2f} +/- {exp_record[1,-1]:.2f}%\n')
            f.write(f'Avg Time: {str(datetime.timedelta(seconds=int(exp_record[2,-2])))} +/- {str(datetime.timedelta(seconds=int(exp_record[2,-1])))}\n')
            f.write(f'Params: {n_parameters / 1.e6:.2f}M\n')
            f.write('='*80 + '\n')
        
        print('\n' + '='*80)
        print('FINAL RESULTS')
        print('='*80)
        print(f'Last Acc1: {exp_record[0,-2]:.2f} +/- {exp_record[0,-1]:.2f}%')
        print(f'Max Acc1: {exp_record[1,-2]:.2f} +/- {exp_record[1,-1]:.2f}%')
        print(f'Avg Time: {str(datetime.timedelta(seconds=int(exp_record[2,-2])))} +/- {str(datetime.timedelta(seconds=int(exp_record[2,-1])))}')
        print(f'Params: {n_parameters / 1.e6:.2f}M')
        print('='*80)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)