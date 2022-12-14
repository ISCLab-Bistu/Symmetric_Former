# SF for Object Detection

This repo contains the supported code and configuration files to reproduce object detection results of "SFormer". It is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Results and Models

### Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | #params |                                          config                                          |
|:--------:| :---: | :---: |:-------:|:-------:|:----------------------------------------------------------------------------------------:|
|    SF    | ImageNet-1K | 1x |  37.4   |  23.2M  | [config](configs/SF/mask_rcnn_sf_patch4_window7_mstrain_480-800_adamw_1x_coco.py) |

### RetinaNet

| Backbone | Pretrain | Lr Schd | box mAP | #params |                                          config                                          |
|:--------:| :---: | :---: |:-------:|:-------:|:----------------------------------------------------------------------------------------:|
|    SF    | ImageNet-1K | 1x |  35.8   |  12.7M  | [config](configs/SF/retina_sf_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py) |

**Notes**: 

- **Pre-trained models can be downloaded from [Symmetric_Former_classification](../classification)**.

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train a Mask R-CNN model with a `SF` backbone and 2 gpus, run:
```
tools/dist_train.sh configs/SF/mask_rcnn_sf_patch4_window7_mstrain_480-800_adamw_1x_coco.py 2 --cfg-options model.pretrained=<PRETRAIN_MODEL> 
```

**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.


### Apex (optional):
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the [configuration files](configs/SF):
```
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

[//]: # (## Citing Swin Transformer)

[//]: # (```)

[//]: # (@article{liu2021Swin,)

[//]: # (  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},)

[//]: # (  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},)

[//]: # (  journal={arXiv preprint arXiv:2103.14030},)

[//]: # (  year={2021})

[//]: # (})

[//]: # (```)

## Other Links

> **Image Classification**: See [SF for classification](../classification).

> **Semantic Segmentation**: See [SF for Semantic Segmentation](../sengmentation).

