"""
bdd_helper.py
    BengaluruDepthDatasetIterator
    BengaluruOccupancyDatasetIterator
"""

import itertools
import os

import cv2
import numpy as np
import pandas as pd
import yaml

DEFAULT_DATSET_BASE = os.path.expanduser("~/Datasets/Depth_Dataset_Bengaluru")
DEFAULT_DATASET = os.path.join(DEFAULT_DATSET_BASE, "1653972957447")
DEFAULT_CALIB = os.path.join(
    DEFAULT_DATSET_BASE, "calibration/pocoX3/calib.yaml"
)


class BengaluruDepthDatasetIterator:
    def __init__(
        self,
        dataset_path=DEFAULT_DATASET,
        settings_doc=DEFAULT_CALIB,
        **kwargs,
    ) -> None:
        self.dataset_path = os.path.expanduser(dataset_path)
        self.dataset_id = self.dataset_path.split("/")[-1]
        self.rgb_img_folder = os.path.join(self.dataset_path, "rgb_img")
        self.depth_img_folder = os.path.join(self.dataset_path, "depth_img")
        self.seg_img_folder = os.path.join(self.dataset_path, "seg_img")
        self.csv_path = os.path.join(
            self.dataset_path, self.dataset_id + ".csv"
        )

        os.path.isdir(self.dataset_path)
        os.path.isdir(self.rgb_img_folder)
        os.path.isdir(self.depth_img_folder)
        os.path.isdir(self.seg_img_folder)
        os.path.isfile(self.csv_path)

        self.settings_doc = os.path.expanduser(settings_doc)
        with open(self.settings_doc, "r") as stream:
            try:
                self.cam_settings = yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)
        k1 = self.cam_settings["Camera.k1"]
        k2 = self.cam_settings["Camera.k2"]
        p1 = self.cam_settings["Camera.p1"]
        p2 = self.cam_settings["Camera.p2"]
        k3 = 0
        if "Camera.k3" in self.cam_settings:
            k3 = self.cam_settings["Camera.k3"]
        self.DistCoef = np.array([k1, k2, p1, p2, k3])
        self.intrinsic_matrix = np.array(
            [
                [
                    self.cam_settings["Camera.fx"],
                    0.0,
                    self.cam_settings["Camera.cx"],
                ],
                [
                    0.0,
                    self.cam_settings["Camera.fy"],
                    self.cam_settings["Camera.cy"],
                ],
                [0.0, 0.0, 1.0],
            ]
        )

        self.width = self.cam_settings["Camera.width"]
        self.height = self.cam_settings["Camera.height"]

        self.csv_dat = pd.read_csv(self.csv_path)

    def __iter__(self):
        self.line_no = 0
        return self

    def __next__(self):
        if self.line_no >= self.__len__():
            raise StopIteration
        data = self[self.line_no]
        self.line_no += 1
        return data

    def __len__(self):
        return len(self.csv_dat)

    def __getitem__(self, key):
        if key > len(self):
            raise IndexError("Out of bounds; key=", key)
        csv_frame = self.csv_dat.loc[key]
        timestamp = str(int(csv_frame[1]))
        # timestamp = str(csv_frame[1])
        disparity_frame_path = os.path.join(
            self.depth_img_folder, timestamp + ".png"
        )
        seg_frame_path = os.path.join(self.seg_img_folder, timestamp + ".png")
        rgb_frame_path = os.path.join(self.rgb_img_folder, timestamp + ".png")

        assert os.path.isfile(disparity_frame_path), (
            "File missing " + disparity_frame_path
        )
        assert os.path.isfile(seg_frame_path), "File missing " + seg_frame_path
        assert os.path.isfile(rgb_frame_path), "File missing " + rgb_frame_path

        disparity_frame = cv2.imread(disparity_frame_path)
        seg_frame = cv2.imread(seg_frame_path)
        rgb_frame = cv2.imread(rgb_frame_path)

        disparity_frame = cv2.cvtColor(disparity_frame, cv2.COLOR_BGR2GRAY)

        frame = {
            "rgb_frame": rgb_frame,
            "disparity_frame": disparity_frame,
            "seg_frame": seg_frame,
            "csv_frame": csv_frame,
        }

        for key in csv_frame.keys():
            frame[key] = csv_frame[key]
        return frame


Z_OFFSET = 0.0


class BengaluruOccupancyDatasetIterator(BengaluruDepthDatasetIterator):
    def __init__(
        self,
        grid_size=(
            40.0,
            30.0,
            4.0,
        ),  # (128/grid_scale[0], 128/grid_scale[1], 8/grid_scale[2])
        scale=(10.0, 10.0, 10.0),  # voxels per meter
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.scale = scale
        self.grid_size = grid_size
        self.occupancy_shape = list(
            map(
                lambda ind: int(self.grid_size[ind] * self.scale[ind]),
                range(len(self.grid_size)),
            )
        )

        self.baseline = 1.0
        self.fx = self.intrinsic_matrix[0, 0]
        self.fy = self.intrinsic_matrix[1, 1]
        self.cx = self.intrinsic_matrix[0, 2]
        self.cy = self.intrinsic_matrix[1, 2]
        self.focal_length = self.fx

        self.transformation = np.eye(4, 4)

    def transform_occupancy_grid_to_points(
        self, occupancy_grid, threshold=0.5, skip=3
    ):
        occupancy_grid = occupancy_grid.squeeze()

        def f(xi):
            i, j, k = xi
            x, y, z = [
                (i) * self.grid_size[0] / (self.occupancy_shape[0] / 2),
                (j - self.occupancy_shape[1] / 2)
                * self.grid_size[1]
                / (self.occupancy_shape[1] / 2),
                (k - self.occupancy_shape[2] / 2)
                * self.grid_size[2]
                / (self.occupancy_shape[2] / 2),
            ]
            if occupancy_grid[i, j, k] > threshold:
                z = z - Z_OFFSET
                z, x, y = x, y, z
                return (x, y, z)
            return (0, 0, 0)

        final_points = np.array(
            [
                f(xi)
                for xi in itertools.product(
                    range(0, occupancy_grid.shape[0], skip),
                    range(0, occupancy_grid.shape[1], skip),
                    range(0, occupancy_grid.shape[2], skip),
                )
            ]
        )

        final_points = final_points[
            np.logical_not(
                np.logical_and(
                    final_points[:, 0] == 0,
                    final_points[:, 1] == 0,
                    final_points[:, 2] == 0,
                )
            )
        ]

        final_points = np.array(final_points, dtype=np.float32)
        return final_points

    def transform_points_to_occupancy_grid(self, velodyine_points_orig):
        occupancy_grid = np.zeros(self.occupancy_shape, dtype=np.float32)

        velodyine_points = velodyine_points_orig.copy()

        velodyine_points_camera = []

        for index in range(velodyine_points.shape[0]):
            x, y, z = velodyine_points[index, :]

            x, y, z = z, x, y  # Inverted
            z = z + Z_OFFSET

            if (
                np.isinf(x)
                or np.isinf(y)
                or np.isinf(z)
                or np.isnan(x)
                or np.isnan(y)
                or np.isnan(z)
            ):
                continue

            i, j, k = [
                int((x * self.occupancy_shape[0] // 2) // self.grid_size[0])
                * 2,
                int(
                    (y * self.occupancy_shape[1] // 2) // self.grid_size[1]
                    + self.occupancy_shape[1] // 2
                ),
                int(
                    (z * self.occupancy_shape[2] // 2) // self.grid_size[2]
                    + self.occupancy_shape[2] // 2
                ),
            ]

            if (
                0 < i < self.occupancy_shape[0]
                and 0 < j < self.occupancy_shape[1]
                and 0 < k < self.occupancy_shape[2]
            ):
                velodyine_points_camera.append((x, y, z))
                occupancy_grid[i, j, k] = 1.0

        velodyine_points_camera = np.array(
            velodyine_points_camera, dtype=np.float32
        )

        return {
            "occupancy_grid": occupancy_grid,
            "velodyine_points_camera": velodyine_points_camera,
        }

    def __getitem__(self, key):
        frame = super().__getitem__(key)
        disparity = frame["disparity_frame"].astype(np.float32)
        rgb_frame = cv2.cvtColor(frame["rgb_frame"], cv2.COLOR_BGR2RGB)

        depth = self.baseline * self.focal_length * np.reciprocal(disparity)
        depth[np.isinf(depth)] = self.baseline * self.focal_length
        depth[np.isnan(depth)] = self.baseline * self.focal_length
        depth = depth.astype(np.float32)

        hide_mask = np.zeros((self.height, self.width), dtype=bool)
        hide_mask[
            0 : depth.shape[0] // 2 :,
        ] = True
        depth[hide_mask] = float("inf")

        U, V = np.ix_(
            np.arange(self.height), np.arange(self.width)
        )  # pylint: disable=unbalanced-tuple-unpacking
        Z = depth.copy()

        X = (V - self.cx) * Z / self.fx
        Y = (U - self.cy) * Z / self.fy

        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()

        points = np.array([X, Y, Z]).T

        B, G, R = rgb_frame[:, :, 0], rgb_frame[:, :, 1], rgb_frame[:, :, 2]
        B = B.flatten()
        G = G.flatten()
        R = R.flatten()

        points_colors = np.array([B, G, R]).T / 255.0

        occupancy_grid_data = self.transform_points_to_occupancy_grid(points)

        frame["disparity"] = disparity
        frame["depth"] = depth
        frame["points"] = points
        frame["points_colors"] = points_colors
        frame["occupancy_grid"] = occupancy_grid_data["occupancy_grid"]
        frame["velodyine_points_camera"] = occupancy_grid_data[
            "velodyine_points_camera"
        ]

        return frame
