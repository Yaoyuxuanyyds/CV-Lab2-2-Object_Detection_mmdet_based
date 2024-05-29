<h1 align = "center">Lab2: Object Detection Based on Faster-RCNN and Yolo-v3</h1>



<h2 align = "center">1. 项目总览</h2>



​	**本项目使用 openmmlab 系列 mmdetection 目标检测框架，在 Pascal VOC2012 数据集上训练了双阶段检测模型 Faster R-CNN 和单阶段检测器 Yolo v3。**



- 项目包含两个 mmdetection 配置文件 `faster-rcnn_r50_fpn_amp-1x_coco.py` , `yolov3_mobilenetv2_8xb24-320-300e_coco`。分别是对 Faster-RCNN 和 Yolo v3 模型进行自定义修改所使用的配置文件。
- `work_dirs\` 目录下包含了两个模型训练过程中的日志、完整配置文件、模型权重文件、在样例图片上获得的检测结果图片。
- `mmdetection\` 目录为来自 openmmlab 官方 repo 的 mmdetection 项目文件。并且已经针对当前任务进行了自定义修改。具体修改见后文详细介绍。

- `data\` 目录用来存放数据集。由于数据集过大所有请自行下载相应数据集并处理后放于当前目录下。*也可以直接从存放模型权重的云盘地址下载作者处理后的数据集*。





---

---



<h2 align = "center">2. 项目说明</h2>

### (1) 版本信息

- `python 3.9.19`

- 由于服务器显卡驱动版本限制，使用 `cudatoolkit 10.2`
- `pytorch 1.8.0 (py3.9_cuda10.2_cudnn7.6.5_0)`

- `mmcv 2.0.0rc4`
- `mmdet 3.3.0` ***mmdet 版本之间兼容性较差，使用老版本mmdet运行本项目会有很多报错***



### (2) 数据集准备

- 如果希望在本地进行训练，需要对 VOC 数据集进行格式转换与目录重构。

- 在本项目中，作者使用了 mmdet 提供的数据集格式转换工具 `tools/dataset_converters/pascal_voc.py` 将 VOC2012 的 XML 格式转换成了标准 COCO 数据集的 JSON 格式，并使用脚本 `split_data.py` 进行了数据集目录的重构，按照 **8：1：1** 的比例将数据分成训练集，验证集和测试集。

- 下面展示了 VOC 数据集的目录格式和重构后的 COCO 目录格式：

  <img src="work_dirs\static\image-20240525104734980.png" alt="image-20240525104734980" style="zoom:50%;" /><img src="work_dirs\static\image-20240525104610256.png" alt="image-20240525104610256" style="zoom:50%;" />

- **进行格式转换和重构后，请将 coco 目录放在项目根目录的 `data\`目录下。*（后面的指令样例均与这样的项目结果对应）***



###  (3) 修改 mmdetection 源码



- `mmdetection/mmdet/datasets/coco.py`: 将 `CocoDataset` 类中的 `classes` 和 `palette` 从 coco 的类别和配色修改为 VOC 数据集对应的类别名和配色盘。

- `mmdetection/mmdet/evaluation/functional/class_names.py`: 将 `coco_classes()` 的返回结果同样改为 VOC 的类别名称。

- 如果希望模型直接输出第一阶段 RPN 网络的候选框（Proposal Boxes)， 需要对 `mmdetection\mmdet\models\detectors\two_stage.py` 中的对应部分进行修改，即直接将 `rpn_results_list` 作为输出。如下图：

  <img src="work_dirs\static\image-20240529163249312.png" alt="image-20240529163249312" />

- 如果希望选择模型输出的检测框是否包含标签名和 score, 可以对 `mmdetection\mmdet\visualization\local_visualizer.py` 中的 `draw_texts()`进行修改。

  <img src="work_dirs\static\image-20240529163436634.png" alt="image-20240529163436634" />

​	

### (4) 模型配置文件 *（可按照需要自行修改）*

#### `faster-rcnn_r50_fpn_amp-1x_coco.py` 

#### 基础参数设置
- `BATCH_SIZE = 4`: **每个 GPU 的样本数量。按照 GPU 资源自定义设置。**
- `NUM_WORKERS = 4`: **每个 GPU 的工作线程数量。按照 CPU 核心数自定义设置。**
- `MAX_EPOCH = 20`: **训练的最大迭代次数（epochs）**。

#### 模型配置
- `roi_head`:
  - `bbox_head`: 边界框头（bounding box head），用于分类和回归。`num_classes=20` **表示 VOC2012 数据集有 20 个类别。**
- `test_cfg`:
  - `rcnn`: 训练后测试时的配置。
    - `nms`: **非极大值抑制配置，`iou_threshold=0.3`表示 IoU 阈值为 0.3，`type='nms'`表示使用标准非极大值抑制**。
    - `score_thr=0.05`: **得分阈值，低于该值的检测结果将被忽略**。

#### 学习率自动缩放
- `auto_scale_lr`:
  - `base_batch_size=16`: 基础批量大小。
  - `enable=True`: 启用自动学习率缩放。

#### 优化器配置
- `optim_wrapper`:
  - `type='AmpOptimWrapper'`: **使用 AMP（自动混合精度）优化器封装**。
  - `optimizer`:
    - `type='SGD'`: 使用带动量的随机梯度下降优化器。
    - `lr=0.02`: 基础学习率。
    - `momentum=0.9`: 动量因子。
    - `weight_decay=0.0001`: 权重衰减因子。

#### 学习率配置
- `param_scheduler`: 学习率调度策略。
  - 第一个调度器:
    - `type='LinearLR'`: 线性学习率预热。
    - `start_factor=0.001`: 起始学习率系数。
    - `by_epoch=False`: 按迭代更新预热学习率。
    - `begin=0`, `end=1000`: 预热阶段的迭代次数。
  - 第二个调度器:
    - `type='MultiStepLR'`: 使用多步学习率调度。
    - `by_epoch=True`: 按 epoch 更新学习率。
    - `begin=0`, `end=12`: 学习率调度的 epoch 范围。
    - `milestones=[8, 11]`: 在第 8 和第 11 个 epoch 时进行学习率衰减。
    - `gamma=0.1`: 学习率衰减系数。

#### 训练和测试配置
- `train_cfg`:
  - `type='EpochBasedTrainLoop'`: 基于 epoch 的训练循环。
  - `max_epochs=MAX_EPOCH`: 最大 epoch 数。
  - `val_interval=1`: 验证间隔，每 1 个 epoch 进行一次验证。

#### 数据加载器和数据集配置
- `DATASET_TYPE='CocoDataset'`: 数据集类型为 COCO 格式。
- `DATA_ROOT='data/coco/'`: 数据集的根目录。
- `train_dataloader`:
  - `batch_size=BATCH_SIZE`, `num_workers=NUM_WORKERS`: 每批次的样本数和工作线程数。
  - `dataset`: 训练数据集配置。
    - `ann_file='annotations/instances_train2017.json'`: 训练集的注释文件路径。
    - `data_prefix=dict(img='train2017/')`: 训练集图片的前缀路径。
    - `data_root=DATA_ROOT`: 数据集根目录。
    - `filter_cfg=dict(filter_empty_gt=True, min_size=32)`: 数据过滤配置，过滤掉空标注和小于 32 像素的标注。
  - `sampler=dict(shuffle=True, type='DefaultSampler')`: 数据采样器配置，进行数据打乱。
- `val_dataloader` `test_dataloader`: 与 `train_dataloader` 选项基本相同。

#### 日志配置
- `default_hooks`:
  - `logger=dict(type='LoggerHook', interval=10)`: 日志记录钩子，每 10 个 iteration 记录一次日志。
  - `checkpoint=dict(type='CheckpointHook', interval=1)`: 检查点钩子，每 1 个 epoch 保存一次模型检查点。
- `log_processor`: 日志处理配置。
  - `by_epoch=True`: 按 epoch 处理日志。
  - `type='LogProcessor'`: 日志处理器类型。
  - `window_size=50`: 滑动窗口大小。

#### 可视化配置
- `vis_backends`:
  - `dict(type='TensorboardVisBackend')`: 使用 TensorBoard 进行日志记录。

#### 输出目录设置
- `work_dir='work_dirs/faster_rcnn_r50_fpn_amp_voc12_1'`: 工作目录，用于存储输出结果（如模型检查点、日志等）。



####  `yolov3_mobilenetv2_8xb24-320-300e_coco`

#### 基础参数设置
- `_base_`: 基础配置文件路径，继承自 `mmdetection` 中的 YOLOv3 MobileNetV2 配置。
- `BATCH_SIZE = 48`: **每个 GPU 的样本数量**。
- `NUM_WORKERS = 4`: **每个 GPU 的工作线程数量**。
- `MAX_EPOCH = 30`: **训练的最大迭代次数（epochs）**。

#### 模型配置
- `model`: 模型结构配置。
  - `bbox_head`:
    - `num_classes=20`: **检测的类别数量为 20，对应 VOC 数据集**。
    - `anchor_generator`: 锚框生成器配置，指定不同尺度的锚框大小。
      - `base_sizes`: 不同特征图层对应的锚框大小。
- `input_size = (320, 320)`: 输入图像的大小。

#### 数据加载器和数据集配置
- `dataset_type = 'CocoDataset'`: 数据集类型为 COCO 格式。
- `data_root = 'data/coco/'`: 数据集的根目录。
- `train_dataloader`:
  - `batch_size=BATCH_SIZE`, `num_workers=NUM_WORKERS`: 每批次的样本数和工作线程数。
  - `dataset`: 训练数据集配置。
    - `type='RepeatDataset'`: **使用重复数据集加速训练**。
    - `times=10`: **数据集重复次数**。

#### 模型设置
- `model`:
  - `train_cfg`:
    - `assigner`: 栅格分配器配置。
      - `type='GridAssigner'`: 使用栅格分配器。
      - `pos_iou_thr=0.5`, `neg_iou_thr=0.5`: 正负样本的 IoU 阈值。
      - `min_pos_iou=0`: 最小正样本 IoU。
  - `test_cfg`:
    - `nms_pre=1000`: 在 NMS 前的最大候选框数。
    - `min_bbox_size=0`: 最小边界框大小。
    - `score_thr=0.05`: 得分阈值。
    - `conf_thr=0.005`: 置信度阈值。
    - `nms`: 非极大值抑制配置。
      - `type='nms'`, `iou_threshold=0.45`: 使用标准 NMS，IoU 阈值为 0.45。
    - `max_per_img=100`: 每张图像的最大检测数。

#### 学习率自动缩放
- `auto_scale_lr`:
  - `base_batch_size=192`: 基础批量大小。

#### 优化器配置
- `optim_wrapper`:
  - `type='OptimWrapper'`: 优化器封装类型。
  - `optimizer`:
    - `type='SGD'`: 使用带动量的随机梯度下降优化器。
    - `lr=0.003`: 学习率。
    - `momentum=0.9`: 动量因子。
    - `weight_decay=0.0005`: 权重衰减因子。
  - `clip_grad`: 梯度裁剪配置。
    - `max_norm=35`, `norm_type=2`: 最大梯度范数和范数类型。

##### 其余配置与 Faster-RCNN 配置文件基本相同



---

---



<h2 align = "center">3. 训练与测试</h2>



- ***由于 mmdetection 框架源码属于子模块，请使用 `git clone` 时加上 `--recursive` 参数。它会自动初始化并更新每一个子模块。***

- **在完成了上述数据集准备工作，并按照自己的需求修改 mmdetection 框架脚本与配置文件后，可以在项目根目录下通过下面的指令进行模型的训练与测试。**

- ***如果选择加载作者训练得到的模型权重，请至云盘地址下载 .pth 文件后放于对应模型的 `work_dirs/` 目录下。***

  

### 1. 单 GPU 训练

`train.py` 脚本用于在单 GPU 环境中启动训练任务。

- #### 基本用法

```bash
python tools/train.py <CONFIG> [optional arguments]
```

- #### 参数说明
  - `<CONFIG>`: 配置文件的路径。例如 `configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py`。


- #### 可选参数
  - `--work-dir <WORK_DIR>`: 训练输出的工作目录。如果不指定，将使用配置文件中的 `work_dir`。

  - `--resume-from <RESUME_FROM>`: 从指定的检查点恢复训练。

  - `--no-validate`: 训练过程中不进行验证。

  - `--gpus <GPUS>`: 使用的 GPU 数量，默认是 1，可以设置为其它值以在多 GPU 环境下进行单机多卡训练。

  - `--seed <SEED>`: 设置随机种子以确保实验可重复性。

  - `--deterministic`: 确保训练是确定性的（即，消除所有可能的随机性）。

  - `--launcher <LAUNCHER>`: 启动方式，可选 `none`, `pytorch`, `slurm`, `mpi`，在单 GPU 训练时通常使用 `none`。


- #### 示例用法


​	假设你有一个配置文件路径为 `yolov3_mobilenetv2_8xb24-320-300e_coco.py`，并且你希望使用默认的工作目录进行训练，你可以这样运行：

```bash
python tools/train.py yolov3_mobilenetv2_8xb24-320-300e_coco.py
```

​	如果你希望指定工作目录为 `work_dirs\yolov3_mobilenetv2_voc12`，你可以这样运行：

```bash
python tools/train.py yolov3_mobilenetv2_8xb24-320-300e_coco.py --work-dir work_dirs\yolov3_mobilenetv2_voc12
```

如果你希望从某个检查点恢复训练，检查点路径为 `work_dirs\yolov3_mobilenetv2_voc12/latest.pth`，你可以这样运行：

```bash
python tools/train.py yolov3_mobilenetv2_8xb24-320-300e_coco.py --resume-from work_dirs\yolov3_mobilenetv2_voc12/latest.pth
```



### 2. 多 GPU 分布式训练

`dist_train.sh` 脚本用于在分布式环境中启动训练任务。

- #### 基本用法

```bash
./tools/dist_train.sh <CONFIG> <GPUS> [PORT] [--work-dir <WORK_DIR>] [--resume-from <RESUME_FROM>] [--no-validate]
```

- #### 参数说明
  - `<CONFIG>`: 配置文件的路径。例如 `configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py`。

  - `<GPUS>`: 使用的 GPU 数量。例如 `8`。

  - `[PORT]` (可选): 启动的端口号。如果不指定，默认端口为 `29500`。

  - `[--work-dir <WORK_DIR>]` (可选): 训练输出的工作目录。如果不指定，将使用配置文件中的 `work_dir`。

  - `[--resume-from <RESUME_FROM>]` (可选): 从指定的检查点恢复训练。

  - `[--no-validate]` (可选): 训练过程中不进行验证。


- #### 示例用法

​	假设配置文件路径为 `yolov3_mobilenetv2_8xb24-320-300e_coco.py`，并且希望使用 4 个 GPU 进行训练，同时指定工作目录为 `work_dirs\yolov3_mobilenetv2_voc12`，你可以这样运行：

```bash
mmdetection/tools/dist_train.sh configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py 4 --work-dir work_dirs\yolov3_mobilenetv2_voc12
```

​	如果你希望从某个检查点恢复训练，检查点路径为 `work_dirs\yolov3_mobilenetv2_voc12/latest.pth`，你可以这样运行：

```bash
mmdetection/tools/dist_train.sh yolov3_mobilenetv2_8xb24-320-300e_coco.py 4 --work-dir work_dirs\yolov3_mobilenetv2_voc12 --resume-from work_dirs\yolov3_mobilenetv2_voc12/latest.pth
```



### 3. 使用测试数据集测试模型

`test.py` 脚本用于在单 GPU 或多 GPU 环境中对训练好的模型进行评估。

- #### 基本用法

```bash
python tools/test.py <CONFIG> <CHECKPOINT> [optional arguments]
```

- #### 参数说明
  - `<CONFIG>`: 配置文件的路径。例如 `configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py`。

  - `<CHECKPOINT>`: 训练好的模型的检查点文件路径。例如 `work_dirs/yolov3_mobilenetv2_custom/latest.pth`。


- #### 可选参数
  - `--out <OUT>`: 将测试结果保存到指定文件中。例如 `--out results.pkl`。

  - `--eval <EVAL>`: 评估指标，例如 `bbox`、`segm`，多个指标可以用逗号分隔。例如 `--eval bbox segm`。

  - `--gpu-ids <GPU_IDS>`: 指定使用的 GPU ID，例如 `--gpu-ids 0`。

  - `--show`: 可视化测试结果（在图像上画出检测框）。

  - `--show-dir <SHOW_DIR>`: 将可视化结果保存到指定目录中。
    - `--cfg-options`: 以键值对的形式修改配置文件中的配置项。
    - `--launcher <LAUNCHER>`: 启动方式，可选 `none`, `pytorch`, `slurm`, `mpi`，在单 GPU 评估时通常使用 `none`。
    - `--local_rank <LOCAL_RANK>`: 用于多 GPU 测试时的本地进程编号，一般不需要手动设置。


- #### 示例用法


​	假设你有一个配置文件路径为 `yolov3_mobilenetv2_8xb24-320-300e_coco.py`，以及训练好的模型检查点文件路径为 `work_dirs\yolov3_mobilenetv2_voc12/latest.pth`。

1. **基本测试**
  
   ```bash
   python tools/test.py yolov3_mobilenetv2_8xb24-320-300e_coco.py work_dirs\yolov3_mobilenetv2_voc12/latest.pth
   ```
   
2. **指定评估指标**
   ```bash
   python tools/test.py yolov3_mobilenetv2_8xb24-320-300e_coco.py work_dirs\yolov3_mobilenetv2_voc12/latest.pth --eval bbox
   ```

3. **保存测试结果**
   ```bash
   python tools/test.py yolov3_mobilenetv2_8xb24-320-300e_coco.py work_dirs\yolov3_mobilenetv2_voc12/latest.pth --out results.pkl
   ```

4. **将可视化结果保存到指定目录**
  
   ```bash
   python tools/test.py yolov3_mobilenetv2_8xb24-320-300e_coco.py work_dirs\yolov3_mobilenetv2_voc12/latest.pth --show-dir results
   ```
   



#### 4. 单张 demo 图片测试

`image_demo.py` 脚本用于对单张图像进行推理和可视化。

- #### 基本用法

```bash
python demo/image_demo.py <IMAGE_PATH> <CONFIG> <CHECKPOINT> [optional arguments]
```

- #### 参数说明
  - `<IMAGE_PATH>`: 输入图像的路径。例如 `test.jpg`。

  - `<CONFIG>`: 配置文件的路径。例如 `configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py`。

  - `<CHECKPOINT>`: 训练好的模型的检查点文件路径。例如 `work_dirs/yolov3_mobilenetv2_custom/latest.pth`。


- #### 可选参数
  - `--device <DEVICE>`: 指定使用的设备，例如 `cuda:0` 或 `cpu`。默认是 `cuda:0`。

  - `--score-thr <SCORE_THR>`: 显示检测结果的分数阈值。默认值为 `0.3`。


- #### 示例用法


​	假设你有一个图像文件路径为 `test.jpg`，一个配置文件路径为 `yolov3_mobilenetv2_8xb24-320-300e_coco.py`，以及训练好的模型检查点文件路径为 `work_dirs\yolov3_mobilenetv2_voc12/latest.pth`。

1. **基本推理**
  
   ```bash
   python demo/image_demo.py test.jpg yolov3_mobilenetv2_8xb24-320-300e_coco.py work_dirs\yolov3_mobilenetv2_voc12/latest.pth
   ```
   
2. **在指定设备上进行推理**
  
   ```bash
   python demo/image_demo.py test.jpg yolov3_mobilenetv2_8xb24-320-300e_coco.py work_dirs\yolov3_mobilenetv2_voc12/latest.pth --device cpu(cuda:0)
   ```
   
   





