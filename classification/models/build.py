# --------------------------------------------------------
# SF Transformer
# Copyright (c) 2022 ISCLab-Bistu
# Licensed under The MIT License [see LICENSE for details]
# Written by Shengjun Liang
# --------------------------------------------------------

from .sf import SF


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'SF':
        model = SF(img_size=config.DATA.IMG_SIZE,
                   patch_size=config.MODEL.SF.PATCH_SIZE,
                   in_chans=config.MODEL.SF.IN_CHANS,
                   num_classes=config.MODEL.NUM_CLASSES,
                   embed_dim=config.MODEL.SF.EMBED_DIM,
                   depths=config.MODEL.SF.DEPTHS,
                   num_heads=config.MODEL.SF.NUM_HEADS,
                   window_size=config.MODEL.SF.WINDOW_SIZE,
                   mlp_ratio=config.MODEL.SF.MLP_RATIO,
                   qkv_bias=config.MODEL.SF.QKV_BIAS,
                   qk_scale=config.MODEL.SF.QK_SCALE,
                   drop_rate=config.MODEL.DROP_RATE,
                   drop_path_rate=config.MODEL.DROP_PATH_RATE,
                   ape=config.MODEL.SF.APE,
                   patch_norm=config.MODEL.SF.PATCH_NORM,
                   use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
