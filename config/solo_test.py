
workdir = 'workdir'

img_norm_cfg = dict(mean=(123.675, 116.280, 103.530), std=(58.395, 57.120, 57.375))
ignore_label = 255

dataset_type = 'CocoDataset'
dataset_root = 'data/coco/'

data = dict(
    val=dict(
        dataset=dict(
            type=dataset_type,
            ann_file=dataset_root + 'annotations/instances_val2017.json',
            img_prefix=dataset_root + 'val2017/',
        ),
        transforms=[
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            # dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(1344, 800)),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_bboxes_ignore'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip')
            ),
        ],
        loader=dict(
            type='DataLoader',
            batch_size=2,
            num_workers=4,
            shuffle=False,
            drop_last=True,
        ),
    )
)

num_outs = 5
grid_numbers = [40, 36, 24, 16, 12]
strides = [8, 8, 16, 32, 32]

model = dict(
    type='SOLO',
    backbone=dict(
        type='ResNet',
        arch='resnet50',
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
        num_classes=80,
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
        scales=[[0, 96], [48, 192], [96, 384], [192, 768], [384, -1]],
        inner_thres=0.2
    )
)
