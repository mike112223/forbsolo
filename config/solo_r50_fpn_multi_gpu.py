
workdir = 'workdir'
gpu_id = '2,3'
num_gpu = len(gpu_id.split(','))


seed = 123
deterministic = True

img_norm_cfg = dict(mean=(123.675, 116.280, 103.530), std=(58.395, 57.120, 57.375))
ignore_label = 255

dataset_type = 'CocoDataset'
dataset_root = 'data/coco/'

data = dict(
    train=dict(
        dataset=dict(
            type=dataset_type,
            ann_file=dataset_root + 'annotations/instances_val2017.json',
            img_prefix=dataset_root + 'val2017/',
        ),
        transforms=[
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_bboxes_ignore'],
                # meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip')
            ),
        ],
        loader=dict(
            type='BaseDataLoader',
            sampler=dict(
                type='GroupSampler',
            ),
            samples_per_gpu=2,
            num_gpu=num_gpu,
            num_workers=num_gpu,
            drop_last=True,
        ),
    ),

    val=dict(
        dataset=dict(
            type=dataset_type,
            ann_file=dataset_root + 'annotations/instances_val2017.json',
            img_prefix=dataset_root + 'val2017/',
        ),
        transforms=[
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_bboxes_ignore'],
                # meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip')
            ),
        ],
        loader=dict(
            type='BaseDataLoader',
            sampler=dict(
                type='GroupSampler',
            ),
            samples_per_gpu=4,
            num_gpu=num_gpu,
            num_workers=num_gpu,
            drop_last=True,
        ),
    )

)

num_outs = 5
num_classes = 80
grid_numbers = [40, 36, 24, 16, 12]
strides = [8, 8, 16, 32, 32]

model = dict(
    type='SOLO',
    backbone=dict(
        type='ResNet',
        arch='resnet50',
        frozen_stages=1,
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=num_outs
    ),
    head=dict(
        type='SOLOHead',
        num_classes=num_classes,
        num_inputs=num_outs,
        in_channels=256,
        feat_channels=256,
        grid_numbers=grid_numbers,
        strides=strides,
        stacked_convs=7,
    ),
    grid=dict(
        type='SOLOGrid',
        grid_numbers=grid_numbers,
        strides=strides,
        scales=[[0, 96], [48, 192], [96, 384], [192, 768], [384, 2048]],
        inner_thres=0.2
    ),
    criterion=dict(
        type='Criterion',
        cls_loss=dict(
            type='FocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        seg_loss=dict(
            type='DiceLoss',
            loss_weight=3.0
        ),
        num_classes=num_classes
    )
)


optim = dict(
    optimizer=dict(
        type='SGD',
        # num gpus
        lr=0.01 * num_gpu / 8,
        momentum=0.9,
        weight_decay=0.0001
    ),
    lr_scheduler=dict(
        type='StepLR',
        lr_step=[9, 11],
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=1.0 / 3,
    )
)


runner = dict(
    type='Runner',
    max_epochs=12,
    trainval_ratio=1,
    snapshot_interval=5,
    print_freq=20,
)
