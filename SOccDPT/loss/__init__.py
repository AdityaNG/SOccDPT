import numpy as np


def color_segmentation(disp_img_masks, frame, class_2_color):
    disp_img_masks_bool = disp_img_masks > 0.5
    disp_img = np.zeros_like(frame)
    for class_index in range(disp_img_masks_bool.shape[2]):
        class_mask = disp_img_masks_bool[:, :, class_index]
        class_color = class_2_color[class_index]
        # assign color to mask
        disp_img[class_mask] = class_color
    return disp_img


def freeze_pretrained_encoder(model):
    for param in model.pretrained.parameters():
        param.requires_grad = False


def unfreeze_pretrained_encoder_by_percentage(model, percentage):
    assert 0 <= percentage <= 1, "percentage must be between 0 and 1"

    parameters = list(model.pretrained.parameters())
    N = len(parameters)
    M = round(N * percentage)
    unfreeze_indices = range(0, M, 1)
    for index, param in enumerate(model.pretrained.parameters()):
        if index in unfreeze_indices:
            param.requires_grad = True
        else:
            param.requires_grad = False
