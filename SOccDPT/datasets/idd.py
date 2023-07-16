import os

import torch
from torch.utils.data import ConcatDataset, Dataset

from .anue_labels import (
    IDD_DATASET_PATH,
    IDD_Dataset,
    get_train_val_test_folders,
    level1_to_class,
)

DEFAULT_DATASET_BASE = os.path.expanduser("~/Datasets/IDD_Segmentation/")


default_leftImg8bit_path = os.path.join(
    DEFAULT_DATASET_BASE, "leftImg8bit/train/0/"
)
default_gtFine_path = os.path.join(DEFAULT_DATASET_BASE, "gtFine/train/0/")
default_depth_path = os.path.join(DEFAULT_DATASET_BASE, "depth/train/0/")


class IDD_Segmentation(Dataset):
    """
    Bengaluru Depth Dataset
        RGB data
        Boosted depth
    """

    def __init__(
        self,
        leftImg8bit_path=default_leftImg8bit_path,
        gtFine_path=default_gtFine_path,
        depth_path=default_depth_path,
        level_id="level1Ids",
        level_2_class=level1_to_class,
        transform=None,
    ):
        super().__init__()
        self.idd = IDD_Dataset(
            leftImg8bit_path=leftImg8bit_path,
            gtFine_path=gtFine_path,
            depth_path=depth_path,
            level_id=level_id,
            level_2_class=level_2_class,
        )
        assert transform is not None
        self.img_transform = transform

    def __len__(self):
        return len(self.idd)

    def __getitem__(self, frame_index):
        # leftImg8bit, seg_map, depth = self.idd[frame_index]
        rgb_frame, seg_frame_bool, depth = self.idd[frame_index]

        # rgb_frame = leftImg8bit
        # seg_frame_bool = seg_map

        x = self.img_transform({"image": rgb_frame})["image"]
        x = torch.from_numpy(x).unsqueeze(0)
        x_raw = torch.tensor(rgb_frame).unsqueeze(
            0
        )  # Image used for visualization
        y = torch.tensor(seg_frame_bool).unsqueeze(0)
        y = y.permute(0, 3, 1, 2)  # Channels first
        mask = torch.ones_like(y, dtype=torch.bool)

        return [x, x_raw, mask, y]


def get_all_IDD_Segmentation_datasets(
    transform,
    level_id="level1Ids",
    level_2_class=level1_to_class,
):
    # train_folders, val_folders, test_folders = get_train_val_test_folders()
    train_folders, val_folders, _ = get_train_val_test_folders()
    train_datasets = []
    val_datasets = []
    # test_datasets = []

    for folder in train_folders:
        train_datasets.append(
            IDD_Segmentation(
                leftImg8bit_path=os.path.join(
                    IDD_DATASET_PATH, "leftImg8bit", "train", folder
                ),
                gtFine_path=os.path.join(
                    IDD_DATASET_PATH, "gtFine", "train", folder
                ),
                depth_path=os.path.join(
                    IDD_DATASET_PATH, "depth", "train", folder
                ),
                transform=transform,
                level_id=level_id,
                level_2_class=level_2_class,
            )
        )

    for folder in val_folders:
        val_datasets.append(
            IDD_Segmentation(
                leftImg8bit_path=os.path.join(
                    IDD_DATASET_PATH, "leftImg8bit", "val", folder
                ),
                gtFine_path=os.path.join(
                    IDD_DATASET_PATH, "gtFine", "val", folder
                ),
                depth_path=os.path.join(
                    IDD_DATASET_PATH, "depth", "val", folder
                ),
                transform=transform,
                level_id=level_id,
                level_2_class=level_2_class,
            )
        )

    train_datasets = ConcatDataset(train_datasets)
    val_datasets = ConcatDataset(val_datasets)

    return train_datasets, val_datasets


if __name__ == "__main__":
    train_datasets, val_datasets = get_all_IDD_Segmentation_datasets(
        transform=lambda x: x
    )

    print("len(train_datasets)", len(train_datasets))
    print("len(val_datasets)", len(val_datasets))
