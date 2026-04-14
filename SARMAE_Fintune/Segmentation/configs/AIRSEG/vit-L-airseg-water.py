############################### AIRSEG合并数据集配置 #################################

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

############################### dataset #################################

dataset_type = 'AIRSEGDataset_Water'
data_root = './Raw_AIR-PolarSAR-Seg'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=1,  
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train_set/images', 
            seg_map_path='train_set_water/annotations'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test_set/images', 
            seg_map_path='test_set_water/annotations'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test_set/images', 
            seg_map_path='test_set_water/annotations'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

############################### running schedule #################################

optim_wrapper = dict(
    type='AmpOptimWrapper',  
    optimizer=dict(
        type='AdamW', lr=4e-5, betas=(0.9, 0.999), weight_decay=0.05),  
    constructor='LayerDecayOptimizerConstructor_ViT', 
    paramwise_cfg=dict(
        num_layers=24,  
        layer_decay_rate=0.9,
    ),
    loss_scale='dynamic'
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),  
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,
        T_max=78500,
        begin=1500,
        end=80000,  
        by_epoch=False,
    )
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=5000) 
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))

############################### model #################################

norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[127.5, 127.5, 127.5],  
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='VisionTransformer_timm',
        img_size=512,
        patch_size=16,
        drop_path_rate=0.2,  
        out_indices=[5, 11, 17, 23], 
        embed_dim=1024,  
        depth=24,  
        num_heads=16,  
        mlp_ratio=4,
        qkv_bias=True,
        use_checkpoint=False,
        pretrained='./sar_change_vitL_200.pth',  
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],  
        num_classes=2,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[0.3, 3.0]  
            ),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                activate=True,
                reduction='mean',
                naive_dice=False,
                loss_weight=1.5,  
                ignore_index=None
            )
        ]
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,  
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=[0.3, 3.0]  
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide',
        stride=(341, 341), 
        crop_size=(512, 512)
    )
)

work_dir = './work_dirs/vit-L-airseg-water'
