# Copyright (c) 2018-2019 Open-MMLab.
# Copyright (c) 2021-2022 Alibaba Group Holding Limited.
# Origin file:
# https://github.com/implus/GFocalV2/blob/master/configs/gfocal/gfocal_r50_fpn_ms2x.py

_base_ = [
    # '../_base_/models/gfocal_r50_fpn_ms6x.py',
    # '../_base_/datasets/coco_detection.py',
    '../_base_/data/visDrone.py'
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# _base_是一个长度为4的list：模型、数据集、优化器、训练方式；Config类作用：将配置文件中的字段转为字典形式
# cfg_text是遍历所有base文件，然后读取其文本内容，并且在前面追加文件路径
# coco_detection.py  文件绝对路径
# coco_detection.py  文件内容

model = dict(
    type='GFL',
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFocalHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=False,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        reg_topk=4,
        reg_channels=64,
        add_mean=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)))
# model training and testing settings
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)

# multi-scale training
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 960)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline))

# Modify dataset related settings
# dataset_type = 'CocoDataset'
dataset_type = 'VisDroneDataset'   # 数据集类型
# data_root = 'data/coco/'
data_root = 'data/visDrone'        # 数据集根目录
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,     # 每个GPU分配的线程数
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/visDrone2019_train.json',
        img_prefix=data_root + 'VisDrone2019-Det-train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/visDrone2019_val.json',
        img_prefix=data_root + 'VisDrone2019-Det-val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/visDrone2019_test.json',
        img_prefix=data_root + 'VisDrone2019-Det-test/images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')    # 评价指标

# optimizer 优化参数，lr为学习率，momentum为动量因子，weight_decay为权重衰减因子
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(
    warmup='linear',        # 初始的学习率增加的策略，linear为线性增加
    warmup_iters=7330,      # 在初始的7330次迭代中学习率逐渐增加
    warmup_ratio=0.1,       # 起始的学习率
    step=[65, 71])          # 每1个epoch存储一次模型
runner = dict(type='EpochBasedRunner', max_epochs=73)
