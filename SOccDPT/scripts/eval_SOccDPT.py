import os

import torch
from torch.utils.data import random_split

from ..utils import (
    evaluate_depth,
    evaluate_seg,
)

from ..datasets.bengaluru_driving_dataset import (
    BDD_Depth_Segmentation,
    class_2_color as class_2_color_bdd,
    get_bdd_dataset,
)
from ..datasets.idd import (
    get_all_IDD_Depth_Segmentation_datasets,
)
from ..datasets.anue_labels import (
    # LEVEL1_ID,
    # level1_to_class,
    # level1_to_color as class_2_color_idd,

    LEVEL4_BASICS_ID,
    level4_basics_to_class,
    level4_basics_to_color as class_2_color_idd,
)
from ..model.loader import load_model, load_transforms
from ..model.SOccDPT import SOccDPT_versions, model_types
from ..model.SOccDPT import DepthNet , SegNet
from ..utils import (
    blockPrint, enablePrint
)

import numpy as np
import random
import cv2
from tqdm import tqdm
import time


@torch.no_grad()
def main(args):
    enablePrint()
    print(f"Model: SOccDPT_V{str(args.version)}_{args.model_type}")
    blockPrint()
    SOccDPT_version = args.version
    SOccDPT = SOccDPT_versions[SOccDPT_version]
    device = torch.device(args.device)
    model_type = args.model_type
    dataset_name = args.dataset
    base_path = args.base_path
    load_seg = args.load_seg
    load_depth = args.load_depth
    load = args.load
    device_cpu = torch.device("cpu")

    transforms, net_w, net_h = load_transforms(
        model_type=model_type,
    )
    dataset = []
    if "idd" in dataset_name:
        train_datasets, val_datasets = get_all_IDD_Depth_Segmentation_datasets(
            transforms,
            level_id=LEVEL4_BASICS_ID,
            level_2_class=level4_basics_to_class,
            # idd_dataset_path=IDD_DATASET_PATH,
        )
        dataset = train_datasets + val_datasets
        classes = set(level4_basics_to_class.values())
        num_classes = len(classes)
        class_2_color = class_2_color_idd
    elif "bdd" in dataset_name:
        dataset = get_bdd_dataset(
            BDD_Depth_Segmentation, transforms, base_path
        )
        num_classes = 3
        class_2_color = class_2_color_bdd

    # Load net
    model_kwargs = dict(
        num_classes=num_classes,
        # point_compute_method='numpy',
    )
    if SOccDPT_version == 1:
        model_kwargs["load_depth"] = load_depth
        model_kwargs["load_seg"] = load_seg
    elif SOccDPT_version == 2:
        assert (
            load_depth is None or load_depth is False
        ), "V2 does not support loading depth"
        assert (
            load_seg is None or load_seg is False
        ), "V2 does not support loading seg"
    elif SOccDPT_version == 3:
        model_kwargs["load_depth"] = load_depth
        assert (
            load_seg is None or load_seg is False
        ), "V3 does not support loading seg"

    net = load_model(
        arch=SOccDPT,
        model_kwargs=model_kwargs,
        device=device_cpu,
        model_path=load,
    )

    net = net.to(device=device)

    enablePrint()
    print(
        "Model Parameters: {:.0f}M".format(
            sum(p.numel() for p in net.parameters()) / 1e6
        )
    )
    blockPrint()

    if args.compile:
        net = torch.compile(net)

    # Set all randomgen seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Select a random set of N images
    N = 10
    total_size = len(dataset)
    total_use = N
    total_discard = total_size - total_use
    dataset, _ = random_split(
        dataset,
        [total_use, total_discard],
        generator=torch.Generator().manual_seed(0),
    )
    # dataset = dataset[:10]

    """
    Save visuals in the format
    /media/visuals
        /{model_type}_{dataset_name}_{version}
            /RGB

            /GT_Depth

            /GT_Seg

            /Pred_Depth

            /Pred_Seg
    """

    # Make dirs if they don't exist
    visual_path = os.path.join(
        "media/visuals",
        f"{model_type}_{dataset_name}_{SOccDPT_version}",
    )
    os.makedirs(visual_path, exist_ok=True)

    rgb_path = os.path.join(visual_path, "RGB")
    os.makedirs(rgb_path, exist_ok=True)

    gt_depth_path = os.path.join(visual_path, "GT_Depth")
    os.makedirs(gt_depth_path, exist_ok=True)

    gt_seg_path = os.path.join(visual_path, "GT_Seg")
    os.makedirs(gt_seg_path, exist_ok=True)

    pred_depth_path = os.path.join(visual_path, "Pred_Depth")
    os.makedirs(pred_depth_path, exist_ok=True)

    pred_seg_path = os.path.join(visual_path, "Pred_Seg")
    os.makedirs(pred_seg_path, exist_ok=True)

    for index, batch in tqdm(enumerate(dataset), total=len(dataset)):
        x, x_raw, mask_disp, y_disp, mask_seg, y_seg = batch
        x = x.to(device=device, dtype=torch.float32)
        y_disp = y_disp.to(device=device, dtype=torch.float32)
        y_seg = y_seg.to(device=device, dtype=torch.float32)
        mask_disp = mask_disp.to(device=device, dtype=torch.bool)
        mask_seg = mask_seg.to(device=device, dtype=torch.bool)

        y_disp_pred, y_seg_pred, points = net(x)

        # File name 000x.png
        file_name = f"{index:04d}.png"

        # Save RGB
        frame_rgb = x_raw.numpy()[0]
        frame_rgb_path = os.path.join(rgb_path, file_name)
        cv2.imwrite(frame_rgb_path, frame_rgb)

        # Save GT Depth
        frame_disp = y_disp.numpy()[0]
        frame_disp = (frame_disp - np.min(frame_disp)) / (
            np.max(frame_disp) - np.min(frame_disp)
        )
        frame_disp = cv2.applyColorMap(
            (frame_disp * 255).astype(np.uint8), cv2.COLORMAP_PLASMA
        )
        frame_disp_path = os.path.join(gt_depth_path, file_name)
        cv2.imwrite(frame_disp_path, frame_disp)

        # Save GT Seg
        disp_img_masks = (
            # y_seg_pred[0].permute(1, 2, 0).cpu().numpy()
            y_seg_pred.permute(1, 2, 0).cpu().numpy()
        )
        disp_img_masks_bool = disp_img_masks > 0.5
        disp_img = np.zeros_like(frame_rgb)
        for class_index in range(disp_img_masks_bool.shape[2]):
            class_mask = disp_img_masks_bool[:, :, class_index]
            class_color = class_2_color[class_index]
            # assign color to mask
            disp_img[class_mask] = class_color
        frame_seg_path = os.path.join(gt_seg_path, file_name)
        cv2.imwrite(frame_seg_path, disp_img)
        ####################################################

        # Save Pred Depth
        disp_img = y_disp_pred[0].cpu().numpy()
        disp_img = (disp_img - np.min(disp_img)) / (
            np.max(disp_img) - np.min(disp_img)
        )
        disp_img = cv2.applyColorMap(
            (disp_img * 255).astype(np.uint8), cv2.COLORMAP_PLASMA
        )
        frame_disp_path = os.path.join(pred_depth_path, file_name)
        cv2.imwrite(frame_disp_path, disp_img)

        # Save Pred Seg
        disp_img_masks = (
            # y_seg_pred[0].permute(1, 2, 0).cpu().numpy()
            y_seg_pred.permute(1, 2, 0).cpu().numpy()
        )
        disp_img_masks_bool = disp_img_masks > 0.5
        disp_img = np.zeros_like(frame_rgb)
        for class_index in range(disp_img_masks_bool.shape[2]):
            class_mask = disp_img_masks_bool[:, :, class_index]
            class_color = class_2_color[class_index]
            # assign color to mask
            disp_img[class_mask] = class_color
        frame_seg_path = os.path.join(pred_seg_path, file_name)
        cv2.imwrite(frame_seg_path, disp_img)

    ###################################################
    # Eval Model FPS
    frame_count = 50
    start_time = time.time()
    for _ in range(frame_count):
        _ = net(x)
    end_time = time.time()

    enablePrint()
    print(f"FPS: \
{frame_count / (end_time - start_time):.2f} \
({frame_count} frames in \
{(end_time - start_time):.2f} seconds)"
    )
    blockPrint()
    ###################################################

    disp_wrapper = DepthNet(net)
    seg_wrapper = SegNet(net)
    amp = False
    val_set = dataset

    iou = evaluate_seg(seg_wrapper, val_set, device, amp=amp)
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = evaluate_depth(
        disp_wrapper, val_set, device, amp=amp
    )

    # Print metrics
    enablePrint()
    print(f"IOU: {iou:.4f}")
    print(f"ABS_REL: {abs_rel:.4f}")
    print(f"SQ_REL: {sq_rel:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"RMSE_LOG: {rmse_log:.4f}")
    print(f"A1: {a1:.4f}")
    print(f"A2: {a2:.4f}")
    print(f"A3: {a3:.4f}")
    print("="*20)
    blockPrint()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SOccDPT")
    parser.add_argument(
        "-v",
        "--version",
        choices=[1, 2, 3],
        required=True,
        type=int,
        help="SOccDPT version",
    )

    parser.add_argument(
        "-dt",
        "--dataset",
        choices=["bdd", "idd"],
        required=True,
        help="Dataset to train using",
    )

    parser.add_argument(
        "-t",
        "--model_type",
        choices=model_types,
        required=True,
        help="Model architecture to use",
    )

    parser.add_argument(
        "-d",
        "--device",
        # default="cuda:0" if torch.cuda.is_available() else "cpu",
        default="cpu",
        help="Device to use for training",
    )

    parser.add_argument(
        "-l",
        "--load",
        required=True,
        help="Checkpoint path",
    )

    parser.add_argument(
        "-cm",
        "--compile",
        action="store_true",
        help="Use torch.compile to optimize the model",
    )

    parser.add_argument(
        "-o",
        "--optimize",
        action="store_true",
        help="Optimize the model using net.half",
    )

    parser.add_argument(
        "-b",
        "--base_path",
        default=os.path.expanduser("~/Datasets/Depth_Dataset_Bengaluru"),
        help="Base path to dataset",
    )

    parser.add_argument(
        "-ld",
        "--load_depth",
        default=None,
        help="Which checkpoint to load",
    )

    parser.add_argument(
        "-ls",
        "--load_seg",
        default=None,
        help="Which checkpoint to load",
    )

    args = parser.parse_args()

    main(args=args)
