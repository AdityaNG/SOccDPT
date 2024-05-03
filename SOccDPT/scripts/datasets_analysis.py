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
from ..loss import (
    freeze_pretrained_encoder,
    unfreeze_pretrained_encoder_by_percentage,
)
from ..loss.ssi_loss import ScaleAndShiftInvariantLoss
from ..model.loader import load_model, load_transforms
from ..model.SOccDPT import DepthNet, SegNet, SOccDPT_versions, model_types
from ..patchwise_training import PatchWiseInplace
from ..utils import evaluate, get_batch

from torch.utils.data import random_split
from torch import optim
from tqdm import tqdm
import json
import os
import random
import traceback
from pathlib import Path

import numpy as np
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def datasets_analysis():
    model_type = "dpt_swin2_tiny_256"
    device_cpu = torch.device("cpu")
    base_path = os.path.expanduser("~/Datasets/Depth_Dataset_Bengaluru")
    val_percent = 0.1

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
    dataset_name = "idd"

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
    
    # Print out dataset size and details
    print(f"Dataset Name: {dataset_name}")
    print(f"Number of Training Datasets: {len(train_datasets)}")
    print(f"Number of Validation Datasets: {len(val_datasets)}")
    print(f"Total Dataset Size: {len(dataset)}")
    print(f"Number of Classes: {num_classes}")
    print("="*20)

    dataset_name = "bdd"
    dataset = get_bdd_dataset(
        BDD_Depth_Segmentation, transforms, base_path
    )
    num_classes = 3
    class_2_color = class_2_color_bdd
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    print(f"Dataset Name: {dataset_name}")
    print(f"Number of Training Datasets: {len(train_set)}")
    print(f"Number of Validation Datasets: {len(val_set)}")
    print(f"Total Dataset Size: {len(dataset)}")
    print(f"Number of Classes: {num_classes}")
    print("="*20)


def main():
    datasets_analysis()


if __name__ == "__main__":
    main()