_base_ = 'mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# MMEngine support the following two ways, users can choose
# according to convenience
# optim_wrapper = dict(type='AmpOptimWrapper')
_base_.optim_wrapper.type = 'AmpOptimWrapper'

# 基础参数设置
BATCH_SIZE = 4 # num of samples per gpu
NUM_WORKERS = 4 # num of workers per gpu
MAX_EPOCH = 20 # num of epochs

# 模型配置
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=20) # VOC2012 有 20 个类别
    ), 
    test_cfg=dict(
        rcnn=dict(
            nms=dict(iou_threshold=0.3, type='nms'),
            score_thr=0.05
        )
    )
)  

# 学习率自动缩放
auto_scale_lr = dict(base_batch_size=16, enable=True)

# 优化器配置
optim_wrapper = dict(  
    type='AmpOptimWrapper',  # 优化器封装的类型
    optimizer=dict(  # 优化器配置
        type='SGD',  
        lr=0.02,  # 基础学习率
        momentum=0.9,  # 带动量的随机梯度下降
        weight_decay=0.0001),  # 权重衰减
    clip_grad=None,  # 梯度裁剪的配置
    )

# 学习率配置
param_scheduler = [
    dict(
        type='LinearLR',  # 使用线性学习率预热
        start_factor=0.001, # 学习率预热的系数
        by_epoch=False,  # 按 iteration 更新预热学习率
        begin=0, 
        end=1000), 
    dict(
        type='MultiStepLR',  # 在训练过程中使用 multi step 学习率策略
        by_epoch=True,  # 按 epoch 更新学习率
        begin=0,   
        end=12,  
        milestones=[8, 11],  # 在哪几个 epoch 进行学习率衰减
        gamma=0.1)  # 学习率衰减系数
]

# 训练和测试配置
train_cfg = dict(
    type='EpochBasedTrainLoop',  # 训练循环的类型
    max_epochs=MAX_EPOCH,
    val_interval=1)  # 验证间隔
val_cfg = dict(type='ValLoop')  # 验证循环的类型
test_cfg = dict(type='TestLoop')  # 测试循环的类型


# Dataloader 和 Dataset 配置
DATASET_TYPE = 'CocoDataset'  
DATA_ROOT = 'data/coco/'

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root=DATA_ROOT,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        type=DATASET_TYPE),
    sampler=dict(shuffle=True, type='DefaultSampler'))

val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root=DATA_ROOT,
        test_mode=True,
        type=DATASET_TYPE))

test_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=dict(
        ann_file='annotations/instances_test2017.json',
        backend_args=None,
        data_prefix=dict(img='test2017/'),
        data_root=DATA_ROOT,
        test_mode=True,
        type=DATASET_TYPE))
    



# 日志配置
default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=10),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1))
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)

# 启用 TensorBoard 日志记录
vis_backends = [
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# 输出目录设置
work_dir = 'work_dirs/faster_rcnn_r50_fpn_amp_voc12'

