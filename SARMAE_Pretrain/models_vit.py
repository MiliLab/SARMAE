from typing import Any, Dict

import timm


def _create_timm_model(model_name: str, **kwargs):
    """Create timm model with graceful fallback for unsupported kwargs."""
    create_kwargs: Dict[str, Any] = dict(kwargs)
    try:
        return timm.create_model(model_name, pretrained=False, **create_kwargs)
    except TypeError:
        # Some timm variants do not accept all kwargs (e.g. img_size/global_pool).
        for key in ("img_size", "global_pool"):
            create_kwargs.pop(key, None)
        return timm.create_model(model_name, pretrained=False, **create_kwargs)


def vit_base_patch16(num_classes=1000, img_size=224, drop_path_rate=0.0, global_pool=True):
    global_pool_mode = "avg" if global_pool else "token"
    return _create_timm_model(
        "vit_base_patch16_224",
        num_classes=num_classes,
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool_mode,
    )


def vit_large_patch16(num_classes=1000, img_size=224, drop_path_rate=0.0, global_pool=True):
    global_pool_mode = "avg" if global_pool else "token"
    return _create_timm_model(
        "vit_large_patch16_224",
        num_classes=num_classes,
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool_mode,
    )


def dinov3_vit_small_patch14(num_classes=1000, img_size=224, drop_path_rate=0.0, global_pool=True):
    global_pool_mode = "avg" if global_pool else "token"
    return _create_timm_model(
        "vit_small_patch14_dinov3",
        num_classes=num_classes,
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool_mode,
    )


def dinov3_vit_base_patch14(num_classes=1000, img_size=224, drop_path_rate=0.0, global_pool=True):
    global_pool_mode = "avg" if global_pool else "token"
    return _create_timm_model(
        "vit_base_patch14_dinov3",
        num_classes=num_classes,
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool_mode,
    )


def dinov3_vit_base_patch16(num_classes=1000, img_size=224, drop_path_rate=0.0, global_pool=True):
    global_pool_mode = "avg" if global_pool else "token"
    return _create_timm_model(
        "vit_base_patch16_224",
        num_classes=num_classes,
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool_mode,
    )


def dinov3_vit_large_patch14(num_classes=1000, img_size=224, drop_path_rate=0.0, global_pool=True):
    global_pool_mode = "avg" if global_pool else "token"
    return _create_timm_model(
        "vit_large_patch14_dinov3",
        num_classes=num_classes,
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool_mode,
    )
