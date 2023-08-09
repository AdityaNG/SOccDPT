from ..datasets.bengaluru_driving_dataset import (
    BDD_Depth_Segmentation,
    class_2_color as class_2_color_bdd,
    get_bdd_dataset,
)
from ..datasets.idd import (
    get_all_IDD_Depth_Segmentation_datasets,
)
from ..datasets.anue_labels import (
    LEVEL1_ID,
    level1_to_class,
    level1_to_color as class_2_color_idd,
)
from ..model.loader import load_transforms
from ..model.SOccDPT import model_types

import os
import cv2
import time
import numpy as np
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def run_net(
    load,
    model_type,
    base_path,
    dataset_name,
):
    import onnxruntime as ort

    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                # 'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
        ),
        # 'CPUExecutionProvider',
    ]

    # model_fp32 = load
    model_quant = "{}.dynanic_quant.onnx".format(load)
    # quantized_model = quantize_dynamic(model_fp32, model_quant)

    sess_options = ort.SessionOptions()

    # Set graph optimization level
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # To enable model serialization after graph optimization set this
    # sess_options.optimized_model_filepath = "{}.opt.onnx".format(load)
    sess_options.optimized_model_filepath = model_quant

    net = ort.InferenceSession(load, sess_options, providers=providers)

    # 1. Create dataset
    transforms, _, _ = load_transforms(
        model_type=model_type,
    )
    dataset = []
    if "idd" in dataset_name:
        train_datasets, val_datasets = get_all_IDD_Depth_Segmentation_datasets(
            transforms,
            level_id=LEVEL1_ID,
            level_2_class=level1_to_class,
            # idd_dataset_path=IDD_DATASET_PATH,
        )
        dataset = train_datasets + val_datasets
        # classes = set(level1_to_class.values())
        # num_classes = len(classes)
        class_2_color = class_2_color_idd
    elif "bdd" in dataset_name:
        dataset = get_bdd_dataset(
            BDD_Depth_Segmentation, transforms, base_path
        )
        # num_classes = 3
        class_2_color = class_2_color_bdd

    # Load net

    print("net", type(net))

    x, x_raw, mask_disp, y_disp, mask_seg, y_seg = dataset[0]

    x = x.to(dtype=torch.float32).numpy()
    y_disp = y_disp.to(dtype=torch.float32).numpy()
    y_seg = y_seg.to(dtype=torch.float32).numpy()
    mask_disp = mask_disp.to(dtype=torch.bool).numpy()
    mask_seg = mask_seg.to(dtype=torch.bool).numpy()

    frame_rgb = x_raw.numpy()[0]

    for i in range(10):
        y_disp_pred, y_seg_pred, points = net.run(None, {"input": x})

        y_disp_pred = y_disp_pred[0]
        y_seg_pred = y_seg_pred.transpose((1, 2, 0))
        points = points[0]

        disp_img = y_disp_pred
        disp_img = (disp_img - np.min(disp_img)) / (
            np.max(disp_img) - np.min(disp_img)
        )
        disp_img = cv2.applyColorMap(
            (disp_img * 255).astype(np.uint8), cv2.COLORMAP_PLASMA
        )

        gt_disp_img = torch.tensor(y_disp[0]).detach().squeeze().cpu().numpy()
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

        disp_img_masks = y_seg_pred
        disp_img_masks_bool = disp_img_masks > 0.5
        disp_img = np.zeros_like(frame_rgb)
        for class_index in range(disp_img_masks_bool.shape[2]):
            class_mask = disp_img_masks_bool[:, :, class_index]
            class_color = class_2_color[class_index]
            # assign color to mask
            disp_img[class_mask] = class_color

        gt_disp_img_masks = y_seg[0].transpose(1, 2, 0)
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

        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite("test.png", vis_img)
        # break

    N = 30
    start = time.time()
    for _ in range(N):
        _ = net.run(None, {"input": x})

        # print(res)

    end = time.time()
    print("Time taken: ", (end - start) / N)
    # FPS
    print("FPS: ", 1 / ((end - start) / N))


def main(args):
    dataset_name = args.dataset

    load = args.load
    model_type = args.model_type
    base_path = args.base_path
    dataset_name = args.dataset

    run_net(
        load,
        model_type,
        base_path,
        dataset_name,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SOccDPT")

    parser.add_argument(
        "-t",
        "--model_type",
        choices=model_types,
        required=True,
        help="Model architecture to use",
    )

    parser.add_argument(
        "-dt",
        "--dataset",
        choices=["bdd", "idd", "idd+bdd"],
        required=True,
        help="Dataset to train using",
    )

    parser.add_argument(
        "-l",
        "--load",
        required=True,
        help="Onnx model path",
    )

    parser.add_argument(
        "-b",
        "--base_path",
        default=os.path.expanduser("~/Datasets/Depth_Dataset_Bengaluru"),
        help="Base path to dataset",
    )

    args = parser.parse_args()

    main(args=args)
