from ..datasets.bengaluru_driving_dataset import (
    BDD_Depth_Segmentation,
    # class_2_color as class_2_color_bdd,
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
    # level4_basics_to_color as class_2_color_idd,
)
from ..model.loader import load_model, load_transforms
from ..model.SOccDPT import SOccDPT_versions, model_types

import os
import random

import numpy as np
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def export_net(
    export_path,
    load,
    load_depth,
    load_seg,
    SOccDPT,
    SOccDPT_version,
    device,
    model_type,
    base_path,
    dataset_name,
):
    device_cpu = torch.device("cpu")

    # Clear memory
    torch.cuda.empty_cache()

    # REPRODUCIBILITY
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # 1. Create dataset
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
        # class_2_color = class_2_color_idd
    elif "bdd" in dataset_name:
        dataset = get_bdd_dataset(
            BDD_Depth_Segmentation, transforms, base_path
        )
        num_classes = 3
        # class_2_color = class_2_color_bdd

    # Load net
    model_kwargs = dict(
        num_classes=num_classes,
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
    # net = torch.compile(net) # causes issues

    # Clear memory
    torch.cuda.empty_cache()

    print("net", type(net))

    x, x_raw, mask_disp, y_disp, mask_seg, y_seg = dataset[0]
    x = x.to(device=device, dtype=torch.float32)
    y_disp = y_disp.to(device=device, dtype=torch.float32)
    y_seg = y_seg.to(device=device, dtype=torch.float32)
    mask_disp = mask_disp.to(device=device, dtype=torch.bool)
    mask_seg = mask_seg.to(device=device, dtype=torch.bool)

    x.requires_grad = True

    y_disp_pred, y_seg_pred, points, occupancy_grid = net(x)

    torch.onnx.export(
        # model being run
        net,
        # model input (or a tuple for multiple inputs)
        x,
        # where to save the model (can be a file or file-like object)
        export_path,
        # store the trained parameter weights inside the model file
        export_params=True,
        # the ONNX version to export the model to
        opset_version=13,
        # whether to execute constant folding for optimization
        do_constant_folding=True,
        # the model's input names
        input_names=["input"],
        # the model's output names
        output_names=["output"],
        # variable length axes
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def main(args):
    # Extract dir from args.export_path
    export_dir = os.path.dirname(args.export_path)
    os.makedirs(export_dir, exist_ok=True)

    SOccDPT_version = args.version
    SOccDPT = SOccDPT_versions[SOccDPT_version]

    export_path = args.export_path
    load = args.load
    load_depth = args.load_depth
    load_seg = args.load_seg
    device = torch.device(args.device)
    model_type = args.model_type
    base_path = args.base_path
    dataset_name = args.dataset

    export_net(
        export_path,
        load,
        load_depth,
        load_seg,
        SOccDPT,
        SOccDPT_version,
        device,
        model_type,
        base_path,
        dataset_name,
    )


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
        choices=["bdd", "idd", "idd+bdd"],
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
        default=None,
        help="Which checkpoint to load",
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

    parser.add_argument(
        "-b",
        "--base_path",
        default=os.path.expanduser("~/Datasets/Depth_Dataset_Bengaluru"),
        help="Base path to dataset",
    )

    parser.add_argument(
        "-e",
        "--export_path",
        required=True,  # default=os.path.join('onnx', 'SOccDPT.onnx'),
        help="Which checkpoint to load",
    )

    args = parser.parse_args()

    main(args=args)
