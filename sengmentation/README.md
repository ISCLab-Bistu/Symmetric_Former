# SF for Semantic Segmentaion

This repo contains the supported code and configuration files to reproduce semantic segmentaion results of "SFormer". It is based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.11.0).


### ADE20K

| Backbone | Method | Crop Size | Lr Schd | mIoU | #params | config |
|:--------:| :---: | :---: | :---: |:----:|:-------:| :---: |
| SFormer  | UPerNet | 512x512 | 160K | 37.4 |  6.8M   | [config](configs/sem_fpn/sy_512x512_160k_ade20k.py) |

**Notes**: 
- **Pre-trained models can be downloaded from [Symmetric_Former_classification](../classification)**.
## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE> --eval mIoU

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --eval mIoU

# multi-gpu, multi-scale testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --aug-test --eval mIoU
```

### Training

To train with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
**Notes:** 
- `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.
- The default learning rate and training schedule is for 2 GPUs and 8 imgs/gpu.


## Other Links

> **Image Classification**: See [SF for classification](../classification).

> **Object Detection**: See [SF for Object Detection](../detection).
