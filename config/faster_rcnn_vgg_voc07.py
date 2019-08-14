# ----------------- train setting -----------------
work_dir = 'work_dirs/faster_rcnn_vgg_voc07'
epoch=120
img_batch_size=1
load_from = None

# ----------------- model -----------------
detector = dict(
    type='FasterRCNN',
    num_classes=21
)

# ----------------- data -----------------
data_root = '/home/zzy/Datasets/voc2007/VOCdevkit/VOC2007/'
dataset = dict(
    train=dict(
        type='VOCDataset',
        ann_file=data_root + 'trainval_anno.json',
        img_root=data_root + 'JPEGImages',
        valid_mode=False
    ),
    val=dict(
        type='VOCDataset',
        ann_file=data_root + 'test_anno.json',
        img_root=data_root + 'JPEGImages',
        valid_mode=True
    ),
    test=dict(
        type='VOCDataset',
        ann_file=data_root + 'test_anno.json',
        img_root=data_root + 'JPEGImages',
        valid_mode=True
    )
)

# ----------------- optimizer -----------------
optimizer = dict(
    type='SGD',
    lr=0.01
)

# ----------------- training hook -----------------
lr_hook_cfg=dict(
    policy='step',
    by_epoch=False,
    step=[600000]
)
optimizer_hook_cfg=dict()
checkpoint_hook_cfg=dict(interval=1)
log_hooks_cfg=dict(
    interval=500,
    hooks=[
        dict(type='VOCEvalLoggerHook'),
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ],
)
