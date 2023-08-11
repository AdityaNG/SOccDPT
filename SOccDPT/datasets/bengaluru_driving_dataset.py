import os
from typing import Type, Callable

import cv2
import numpy as np
import torch
import torchvision

from .bdd_helper import (
    BengaluruDepthDatasetIterator,
    BengaluruOccupancyDatasetIterator,
    DEFAULT_CALIB,
    DEFAULT_DATASET,
)


class BDD_Dataset(BengaluruDepthDatasetIterator):
    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET,
        settings_doc: str = DEFAULT_CALIB,
        transform: Callable = lambda x: x,
    ):
        super().__init__(dataset_path=dataset_path, settings_doc=settings_doc)
        assert transform is not None
        self.img_transform = transform

    # def __getitem__(self, frame_index: int):
    #     assert False, "Not implemented"


class BDD_Depth(BDD_Dataset):
    """
    Bengaluru Depth Dataset
        RGB data
        Boosted depth
    """

    def __getitem__(self, frame_index):
        frame = super().__getitem__(frame_index)

        rgb_frame = frame["rgb_frame"]
        disparity_frame = frame["disparity_frame"]
        x = self.img_transform({"image": rgb_frame})["image"]
        x = torch.from_numpy(x).unsqueeze(0)
        x_raw = torch.tensor(rgb_frame).unsqueeze(
            0
        )  # Image used for visualization
        y = torch.tensor(disparity_frame).unsqueeze(0)
        mask = torch.ones_like(y, dtype=torch.bool)

        return [x, x_raw, mask, y]


def normalize(mat):
    return (mat - np.min(mat)) / (np.max(mat) - np.min(mat))


color_2_class = {
    (0, 0, 0): 0,  # Background, black
    (0, 0, 142): 1,  # Vehicle, red
    (220, 20, 60): 2,  # Pedeastrian, blue
}
class_2_color = {v: k for k, v in color_2_class.items()}


def rgb_seg_to_bool(seg_frame):
    seg_frame_bool = np.zeros(
        [seg_frame.shape[0], seg_frame.shape[1], len(color_2_class.keys())],
        dtype=bool,
    )
    for color in color_2_class:
        seg_frame_bool[:, :, color_2_class[color]] = np.all(
            seg_frame == np.array(color), axis=-1
        )
    return seg_frame_bool


class BDD_Segmentation(BDD_Dataset):
    """
    Bengaluru Depth Dataset
        RGB data
        Boosted depth
    """

    def __getitem__(self, frame_index):
        frame = super().__getitem__(frame_index)

        rgb_frame = frame["rgb_frame"]
        seg_frame = frame["seg_frame"]
        seg_frame_bool = rgb_seg_to_bool(seg_frame)
        x = self.img_transform({"image": rgb_frame})["image"]
        x = torch.from_numpy(x).unsqueeze(0)
        x_raw = torch.tensor(rgb_frame).unsqueeze(
            0
        )  # Image used for visualization
        y = torch.tensor(seg_frame_bool).unsqueeze(0)
        y = y.permute(0, 3, 1, 2)  # Channels first
        mask = torch.ones_like(y, dtype=torch.bool)

        return [x, x_raw, mask, y]


class BDD_Depth_Segmentation(BDD_Dataset):
    """
    Bengaluru Depth Dataset
        RGB data
        Boosted depth
    """

    def __getitem__(self, frame_index):
        frame = super().__getitem__(frame_index)

        rgb_frame = frame["rgb_frame"]
        seg_frame = frame["seg_frame"]
        disparity_frame = frame["disparity_frame"]

        # Resize to 1920x1080
        rgb_frame = cv2.resize(rgb_frame, (1920, 1080))
        seg_frame = cv2.resize(seg_frame, (1920, 1080))
        disparity_frame = cv2.resize(disparity_frame, (1920, 1080))

        seg_frame_bool = rgb_seg_to_bool(seg_frame)

        y_disp = torch.tensor(disparity_frame).unsqueeze(0)
        mask_disp = torch.ones_like(y_disp, dtype=torch.bool)

        x = self.img_transform({"image": rgb_frame})["image"]
        x = torch.from_numpy(x).unsqueeze(0)
        x_raw = torch.tensor(rgb_frame).unsqueeze(
            0
        )  # Image used for visualization
        y_seg = torch.tensor(seg_frame_bool).unsqueeze(0)
        y_seg = y_seg.permute(0, 3, 1, 2)  # Channels first
        mask_seg = torch.ones_like(y_seg, dtype=torch.bool)

        return [x, x_raw, mask_disp, y_disp, mask_seg, y_seg]


class BDD_Occupancy_Dataset(BengaluruOccupancyDatasetIterator):
    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET,
        settings_doc: str = DEFAULT_CALIB,
        transform: Callable = lambda x: x,
    ):
        super().__init__(dataset_path=dataset_path, settings_doc=settings_doc)
        assert transform is not None
        self.img_transform = transform

    def __getitem__(self, frame_index):
        frame = super().__getitem__(frame_index)

        rgb_frame = frame["rgb_frame"]
        occupancy_grid = frame["occupancy_grid"]
        # print('occupancy_grid', occupancy_grid.shape)

        # Resize to 1920x1080
        rgb_frame = cv2.resize(rgb_frame, (1920, 1080))

        x = self.img_transform({"image": rgb_frame})["image"]
        x = torch.from_numpy(x).unsqueeze(0)
        x_raw = torch.tensor(rgb_frame).unsqueeze(
            0
        )  # Image used for visualization
        y = torch.from_numpy(occupancy_grid).unsqueeze(0)
        mask = torch.ones_like(y, dtype=torch.bool)

        return [x, x_raw, mask, y]


def get_bdd_dataset(
    BDD_Dataset: Type[BDD_Dataset],
    transform: torchvision.transforms.Compose,
    base_path: str,
) -> torch.utils.data.ConcatDataset:
    # Dataset definition
    dataset = torch.utils.data.ConcatDataset(
        [
            # List of datasets
            BDD_Dataset(
                dataset_path=os.path.join(base_path, "1653972957447"),
                transform=transform,
            ),
            BDD_Dataset(
                dataset_path=os.path.join(base_path, "1652937970859"),
                transform=transform,
            ),
            BDD_Dataset(
                dataset_path=os.path.join(base_path, "1654493684259"),
                transform=transform,
            ),
            BDD_Dataset(
                dataset_path=os.path.join(base_path, "1654507149598"),
                transform=transform,
            ),
            BDD_Dataset(
                dataset_path=os.path.join(base_path, "1658384707877"),
                transform=transform,
            ),
            BDD_Dataset(
                dataset_path=os.path.join(base_path, "1658384924059"),
                transform=transform,
            ),
        ]
    )
    return dataset
