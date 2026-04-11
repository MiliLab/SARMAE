evaluation = dict(interval=1, metric=['bbox', 'segm'], classwise=True)

optimizer = dict(
    type='AdamW', 
    lr=5e-5,  
    betas=(0.9, 0.999), 
    weight_decay=0.1,  
    constructor='LayerDecayOptimizerConstructor_ViT_Old',
    paramwise_cfg=dict(
        num_layers=12, 
        layer_decay_rate=0.95,  
        custom_keys={
            'backbone': dict(lr_mult=0.08, decay_mult=0.5),  
        }
    ))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=800,  
    warmup_ratio=0.3333333333333333,
    step=[22, 30, 34])  
runner = dict(type='EpochBasedRunner', max_epochs=36)  
checkpoint_config = dict(interval=1, max_keep_ckpts=15)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
dataset_type = 'SSDDDataset'
data_root = './det/Official-SSDD-OPEN/BBox_SSDD/coco_style/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='SSDDDataset',
        ann_file=
        './det/Official-SSDD-OPEN/BBox_SSDD/coco_style/annotations/trainval.json',
        img_prefix='./det/Official-SSDD-OPEN/BBox_SSDD/coco_style/images/trainval/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='AutoAugment',
                 policies=[
                     [
                         dict(type='Resize',
                              img_scale=(608, 608),
                              keep_ratio=False)
                     ],
                     [
                         dict(type='Resize',
                              img_scale=[(672, 672), (736, 736), (800, 800)],  
                              multiscale_mode='value',
                              keep_ratio=False),
                         dict(type='RandomCrop',
                              crop_type='absolute_range',
                              crop_size=(560, 672),  
                              allow_negative_crop=True),
                         dict(type='Resize',
                              img_scale=(608, 608),
                              override=True,
                              keep_ratio=False)
                     ]
                 ]),
            dict(type='PhotoMetricDistortion',
                 brightness_delta=35, 
                 contrast_range=(0.5, 1.5),  
                 saturation_range=(0.5, 1.5),  
                 hue_delta=18),  
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(
        type='SSDDDataset',
        ann_file=
        './det/Official-SSDD-OPEN/BBox_SSDD/coco_style/annotations/test.json',
        img_prefix='./det/Official-SSDD-OPEN/BBox_SSDD/coco_style/images/test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(608, 608),
                flip=True,  
                transforms=[
                    dict(type='Resize', keep_ratio=False),  
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='SSDDDataset',
        ann_file=
        './det/Official-SSDD-OPEN/BBox_SSDD/coco_style/annotations/test.json',
        img_prefix='./det/Official-SSDD-OPEN/BBox_SSDD/coco_style/images/test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(608, 608),
                flip=True,  
                transforms=[
                    dict(type='Resize', keep_ratio=False),  
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ])) 
gpu_number = 8
num_classes = 1
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='VisionTransformer_timm',
        img_size=608,
        patch_size=16,
        drop_path_rate=0.3,
        out_indices=[11],  
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        use_checkpoint=False,
        pretrained =  './mae_dinov3_vitb16_pretrain_lvd1689m_timm_format.pth',
        frozen_stages=-1,  
    ),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768, 768],  
        out_channels=256,
        num_outs=5),  
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.001, method='gaussian', sigma=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
fp16 = dict(loss_scale=dict(init_scale=512))
work_dir = './work_dirs/vitb_ssdd'
auto_resume = True
gpu_ids = range(0, 8)
device = 'cuda'

