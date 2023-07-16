from typing import Type

import cv2
import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from ..loss.ssi_loss import compute_scale_and_shift
from ..model.SOccDPT import DepthNet, SegNet, SOccDPT
from ..model.blocks import Interpolate


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


def change_number_of_classes(model, num_classes=150):
    features = model.features

    head = nn.Sequential(
        nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(features),
        nn.ReLU(True),
        nn.Dropout(0.1, False),
        nn.Conv2d(features, num_classes, kernel_size=1),
        Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
    )
    model.scratch.output_conv = head

    model.auxlayer = nn.Sequential(
        nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(features),
        nn.ReLU(True),
        nn.Dropout(0.1, False),
        nn.Conv2d(features, num_classes, kernel_size=1),
    )


def compute_errors(gt, pred):
    """
    Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_masked_errors(gt, pred, mask):
    """
    Computation of error metrics between predicted and ground
    truth depths only for the masked region

    Args:
    gt: numpy array of shape (H, W) representing the ground
        truth depths
    pred: numpy array of shape (H, W) representing the predicted
        depths
    mask: numpy array of shape (H, W) representing the mask indicating
    the region of interest

    Returns:
    tuple of error metrics for the masked region: abs_rel, sq_rel, rmse,
    rmse_log, a1, a2, a3
    """
    masked_gt = gt[mask]
    masked_pred = pred[mask]

    thresh = np.maximum((masked_gt / masked_pred), (masked_pred / masked_gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (masked_gt - masked_pred) ** 2
    rmse = np.sqrt(rmse.mean())
    if np.isinf(rmse) or np.isnan(rmse):
        rmse = 0

    rmse_log = (
        np.log(masked_gt) - np.log(masked_pred)
    ) ** 2  # RuntimeWarning: invalid value encountered in log
    rmse_log = np.sqrt(rmse_log.mean())
    if np.isinf(rmse_log) or np.isnan(rmse_log):
        rmse_log = 0

    abs_rel = np.mean(np.abs(masked_gt - masked_pred) / masked_gt)
    if np.isinf(abs_rel) or np.isnan(abs_rel):
        abs_rel = 0

    sq_rel = np.mean(((masked_gt - masked_pred) ** 2) / masked_gt)
    if np.isinf(sq_rel) or np.isnan(sq_rel):
        sq_rel = 0

    a1 = 0 if np.isnan(a1) else a1
    a2 = 0 if np.isnan(a2) else a2
    a3 = 0 if np.isnan(a3) else a3

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate_depth(net, dataloader, device, amp=False):
    net.eval()
    num_val_batches = len(dataloader)

    abs_rel_l, sq_rel_l, rmse_l, rmse_log_l, a1_l, a2_l, a3_l = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    print("Validation")

    assert len(dataloader) > 0, "Validation set has no elements"
    assert num_val_batches > 0, "num_val_batches set has no elements"
    # iterate over the validation set
    for batch in tqdm(
        dataloader,
        total=num_val_batches,
        desc="Validation round",
        unit="batch",
        leave=False,
    ):
        if len(batch) == 4:
            x, x_raw, mask, y = batch
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.bool)
        else:
            x, x_raw, mask_disp, y_disp, mask_seg, y_seg = batch
            x = x.to(device=device, dtype=torch.float32)
            y = y_disp.to(device=device, dtype=torch.float32)
            y_seg = y_seg.to(device=device, dtype=torch.float32)
            mask = mask_disp.to(device=device, dtype=torch.bool)
            mask_seg = mask_seg.to(device=device, dtype=torch.bool)

        with torch.cuda.amp.autocast(enabled=amp):
            y_pred = net(x)

            if len(y_pred.shape) == 2:
                y_pred = y_pred.unsqueeze(0)

            y_pred = torch.nn.functional.interpolate(
                y_pred.unsqueeze(1),
                size=(y.shape[1], y.shape[2]),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            # if len(y_pred.shape)==2:
            #     y_pred = y_pred.unsqueeze(0)

            # abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(
            #     y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
            # )
            scale, shift = compute_scale_and_shift(y_pred, y, mask)
            y_pred_ssi = scale.view(-1, 1, 1) * y_pred + shift.view(-1, 1, 1)

            (
                abs_rel,
                sq_rel,
                rmse,
                rmse_log,
                a1,
                a2,
                a3,
            ) = compute_masked_errors(
                y.detach().cpu().numpy(),
                y_pred_ssi.detach().cpu().numpy(),
                mask.detach().cpu().numpy(),
            )

            if np.isfinite(abs_rel):
                abs_rel_l += [abs_rel]
            if np.isfinite(sq_rel):
                sq_rel_l += [sq_rel]
            rmse_l += [rmse]
            if np.isfinite(rmse_log):
                rmse_log_l += [rmse_log]
            a1_l += [a1]
            a2_l += [a2]
            a3_l += [a3]

    net.train()

    return (
        np.mean(abs_rel_l),
        np.mean(sq_rel_l),
        np.mean(rmse_l),
        np.mean(rmse_log_l),
        np.mean(a1_l),
        np.mean(a2_l),
        np.mean(a3_l),
    )


def evaluate_seg(net, dataloader, device, amp=False):
    net.eval()
    num_val_batches = len(dataloader)

    iou_l = []

    print("Validation")

    assert len(dataloader) > 0, "Validation set has no elements"
    assert num_val_batches > 0, "num_val_batches set has no elements"
    # iterate over the validation set
    for batch in tqdm(
        dataloader,
        total=num_val_batches,
        desc="Validation round",
        unit="batch",
        leave=False,
    ):
        if len(batch) == 4:
            x, x_raw, mask, y = batch
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.bool)
        else:
            x, x_raw, mask_disp, y_disp, mask_seg, y_seg = batch
            x = x.to(device=device, dtype=torch.float32)
            y_disp = y_disp.to(device=device, dtype=torch.float32)
            y = y_seg.to(device=device, dtype=torch.float32)
            mask_disp = mask_disp.to(device=device, dtype=torch.bool)
            mask = mask_seg.to(device=device, dtype=torch.bool)

        # with torch.cuda.amp.autocast(enabled=amp):
        with torch.no_grad():
            y_pred = net(x)

            if len(y_pred.shape) == 3:
                y_pred = y_pred.unsqueeze(0)

            y_pred = torch.nn.functional.interpolate(
                y_pred,
                # y_pred.unsqueeze(1),
                size=(y.shape[2], y.shape[3]),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            if len(y_pred.shape) == 3:
                y_pred = y_pred.unsqueeze(0)

            num_classes = y_pred.shape[1]
            iou = 0.0

            # Class wise IoU
            for class_id in range(num_classes):
                y_pred_mask = y_pred[:, class_id, :, :] > 0.5
                y_mask = y[:, class_id, :, :] > 0.5

                intersection = torch.logical_and(y_pred_mask, y_mask).sum(
                    dim=(1, 2)
                )
                union = torch.logical_or(y_pred_mask, y_mask).sum(dim=(1, 2))

                iou += intersection / (
                    union + 1e-7
                )  # Adding a small epsilon to avoid division by zero

            iou /= num_classes

            iou_l += [iou.cpu().numpy()]

    net.train()

    return np.mean(iou_l)


##############################################################
# Color map logic

cmap = mpl.colormaps["viridis"]
# cmap = mpl.colormaps['magma']

colors_hash = []
colors_hash_res = 256
for i in range(0, colors_hash_res):
    colors_hash.append(cmap(float(i) / (colors_hash_res - 1)))


def color_by_index(
    POINTS_np, index=2, invert=False, min_height=None, max_height=None
):
    if POINTS_np.shape[0] == 0:
        return np.ones_like(POINTS_np)
    heights = POINTS_np[:, index].copy()
    heights_filter = np.logical_not(
        np.logical_and(np.isnan(heights), np.isinf(heights))
    )
    if max_height is None:
        max_height = np.max(heights[heights_filter])
    if min_height is None:
        min_height = np.min(heights[heights_filter])
    # heights = np.clip(heights, min_height, max_height)
    heights = (heights - min_height) / (max_height - min_height)
    if invert:
        heights = 1.0 - heights
    # heights[np.logical_not(heights_filter)] = 0.0
    heights = np.clip(heights, 0.0, 1.0)
    heights_color_index = np.rint(heights * (colors_hash_res - 1)).astype(
        np.uint8
    )

    COLORS_np = np.array([colors_hash[xi] for xi in heights_color_index])
    return (COLORS_np * 255).astype(np.uint8)


##############################################################
def evaluate(
    net: Type[SOccDPT],
    seg_wrapper: Type[SegNet],
    disp_wrapper: Type[DepthNet],
    val_set,
    device,
    amp,
    x_raw,
    y_disp,
    y_disp_pred,
    y_seg,
    y_seg_pred,
    points,
    class_2_color,
    loss,
    lr,
    global_step,
    epoch,
    experiment,
):
    histograms = {}
    for tag, value in net.named_parameters():
        if value is not None and value.grad is not None:
            tag = tag.replace("/", ".")
            histograms["Weights/" + tag] = wandb.Histogram(value.data.cpu())
            histograms["Gradients/" + tag] = wandb.Histogram(
                value.grad.data.cpu()
            )

    frame_rgb = x_raw[0].detach().squeeze().cpu().numpy()

    ############################################################
    # Eval Disparity
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = evaluate_depth(
        disp_wrapper, val_set, device, amp=amp
    )

    print(
        "abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3",
        abs_rel,
        sq_rel,
        rmse,
        rmse_log,
        a1,
        a2,
        a3,
    )

    disp_img = y_disp_pred[0].detach().squeeze().cpu().numpy()
    disp_img = (disp_img - np.min(disp_img)) / (
        np.max(disp_img) - np.min(disp_img)
    )
    disp_img = cv2.applyColorMap(
        (disp_img * 255).astype(np.uint8), cv2.COLORMAP_PLASMA
    )

    gt_disp_img = y_disp[0].detach().squeeze().cpu().numpy()
    gt_disp_img = (gt_disp_img - np.min(gt_disp_img)) / (
        np.max(gt_disp_img) - np.min(gt_disp_img)
    )
    gt_disp_img = cv2.applyColorMap(
        (gt_disp_img * 255).astype(np.uint8), cv2.COLORMAP_PLASMA
    )

    # resize disp_img to shape of frame_rgb
    disp_img = cv2.resize(disp_img, frame_rgb.shape[:2][::-1])
    gt_disp_img = cv2.resize(gt_disp_img, frame_rgb.shape[:2][::-1])
    vis_img_1 = np.concatenate([frame_rgb, disp_img, gt_disp_img], 1)

    ############################################################
    # Eval Seg
    iou = evaluate_seg(seg_wrapper, val_set, device, amp=amp)

    print("iou", iou)

    disp_img_masks = (
        y_seg_pred[0].permute(1, 2, 0).detach().squeeze().cpu().numpy()
    )
    disp_img_masks_bool = disp_img_masks > 0.5
    disp_img = np.zeros_like(frame_rgb)
    for class_index in range(disp_img_masks_bool.shape[2]):
        class_mask = disp_img_masks_bool[:, :, class_index]
        class_color = class_2_color[class_index]
        # assign color to mask
        disp_img[class_mask] = class_color

    gt_disp_img_masks = (
        y_seg[0].permute(1, 2, 0).detach().squeeze().cpu().numpy()
    )
    gt_disp_img_masks_bool = gt_disp_img_masks > 0.5
    gt_disp_img = np.zeros_like(frame_rgb)
    for class_index in range(gt_disp_img_masks_bool.shape[2]):
        class_mask = gt_disp_img_masks_bool[:, :, class_index]
        class_color = class_2_color[class_index]
        # assign color to mask
        gt_disp_img[class_mask] = class_color

    vis_img_2 = np.concatenate([frame_rgb, disp_img, gt_disp_img], 1)

    ############################################################
    # Visualize image
    vis_img = np.concatenate([vis_img_1, vis_img_2], 0)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    # shrink image
    vis_img = cv2.resize(vis_img, (0, 0), fx=0.5, fy=0.5)

    ############################################################
    # Visualze points
    # plot_points = points[0].permute(1, 2, 0).detach().squeeze().cpu().numpy()
    # if points is not numpy array, convert to numpy array

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    plot_points = points[0]
    plot_points = plot_points.reshape(-1, 3)
    # plot_colors = (gt_disp_img.reshape(-1, 3) / 255.0)
    plot_colors = gt_disp_img.reshape(-1, 3)

    # downsample
    plot_points = plot_points[::10]
    plot_colors = plot_colors[::10]

    # remove points which are black
    black_mask = plot_colors[:, 0] > 0
    plot_points = plot_points[black_mask]
    plot_colors = plot_colors[black_mask]

    plot_points_colors = np.hstack([plot_points, plot_colors])

    print("plot_points_colors", plot_points_colors.shape)

    ############################################################

    print("loss: {}".format(loss))

    experiment.log(
        {
            "learning rate": lr,
            # Disparity metrics
            "abs_rel": abs_rel,
            "sq_rel": sq_rel,
            "rmse": rmse,
            "rmse_log": rmse_log,
            "a1": a1,
            "a2": a2,
            "a3": a3,
            # Segmentation metrics
            "iou": iou,
            "plot": wandb.Image(vis_img),
            "plot_points": wandb.Object3D(
                {
                    "type": "lidar/beta",
                    "points": plot_points_colors,
                }
            ),
            "loss": loss.item(),
            "step": global_step,
            "epoch": epoch,
            **histograms,
        }
    )


def get_batch(train_set, batch_index, batch_size):
    batch = [
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    for sub_index in range(batch_index - batch_size, batch_index, 1):
        new_batch = train_set[sub_index]
        for sub_cat in range(len(batch)):
            batch[sub_cat] += [new_batch[sub_cat]]

    for sub_cat in range(len(batch)):
        batch[sub_cat] = torch.cat(batch[sub_cat], dim=0)

    return batch
