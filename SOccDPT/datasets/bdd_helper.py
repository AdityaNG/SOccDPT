import os
import numpy as np
import yaml
import cv2
import pandas as pd
from PIL import Image
import math


def rgb_seg_to_class(seg_frame, color_2_class):
    seg_frame = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2RGB)
    seg_frame_class = np.zeros(
        [
            seg_frame.shape[0],
            seg_frame.shape[1],
        ],
        dtype=int,
    )

    for color in color_2_class:
        seg_frame_class[
            np.all(seg_frame == np.array(color), axis=-1)
        ] = color_2_class[color]

    return seg_frame_class


def get_item_between_timestamp(csv_dat, start_ts, end_ts, fault_delay=0.5):
    """
    Return frame between two given timestamps
    Raise exception if delta between start_ts and
        minimum_ts is greater than fault_delay
    Raise exception if delta between end_ts and
        maximum_ts is greater than fault_delay
    """
    ts_dat = csv_dat[csv_dat["Timestamp"].between(start_ts, end_ts)]
    minimum_ts = min(ts_dat["Timestamp"])
    if abs(minimum_ts - start_ts) > fault_delay:
        raise Exception("out of bounds: |minimum_ts - start_ts|>fault_delay")
    maximum_ts = max(ts_dat["Timestamp"])
    if abs(maximum_ts - end_ts) > fault_delay:
        raise Exception("out of bounds: |maximum_ts - end_ts|>fault_delay")
    return ts_dat


def parse_rot(rot):
    rot = rot.replace("[", "").replace("]", "").replace("\n", "")
    rot = rot.split()
    rot = np.array(rot).astype(np.float32).reshape(3, 3)
    return rot


DATASET_BASE = "~/Datasets/Depth_Dataset_Bengaluru"
DEFAULT_CALIB = os.path.join(
    DATASET_BASE, "calibration/pocoX3/calib.yaml"
)
DEFAULT_DATASET = os.path.join(DATASET_BASE, "1658384707877")


class BengaluruDepthDatasetIterator:
    def __init__(
        self,
        dataset_path=DEFAULT_DATASET,
        settings_doc=DEFAULT_CALIB,
        file_extension=".png",
    ) -> None:
        self.dataset_path = os.path.expanduser(dataset_path)
        self.dataset_id = self.dataset_path.split("/")[-1]
        self.rgb_img_folder = os.path.join(self.dataset_path, "rgb_img")
        self.depth_img_folder = os.path.join(self.dataset_path, "depth_img")
        self.seg_img_folder = os.path.join(self.dataset_path, "seg_img")
        self.csv_path = os.path.join(
            self.dataset_path, self.dataset_id + ".csv"
        )
        self.traj_path = os.path.join(
            self.dataset_path, self.dataset_id + "_traj.csv"
        )
        self.file_extension = file_extension

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
        self.traj_available = os.path.isfile(self.traj_path)

        if self.traj_available:
            self.traj_dat = pd.read_csv(self.traj_path)
            self.traj_dat["rot"] = self.traj_dat["rot"].apply(parse_rot)

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
        timestamp_ms = int(csv_frame[1])
        timestamp = str(timestamp_ms)

        disparity_frame_path = os.path.join(
            self.depth_img_folder, timestamp + self.file_extension
        )
        seg_frame_path = os.path.join(
            self.seg_img_folder, timestamp + self.file_extension
        )
        rgb_frame_path = os.path.join(
            self.rgb_img_folder, timestamp + self.file_extension
        )

        assert os.path.isfile(disparity_frame_path), (
            "File missing " + disparity_frame_path
        )
        assert os.path.isfile(seg_frame_path), "File missing " + seg_frame_path
        assert os.path.isfile(rgb_frame_path), "File missing " + rgb_frame_path

        rgb_frame = Image.open(rgb_frame_path)
        rgb_frame = np.asarray(rgb_frame)
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        seg_frame = Image.open(seg_frame_path)
        seg_frame = np.asarray(seg_frame)
        seg_frame = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2RGB)

        disparity_frame = Image.open(disparity_frame_path)
        disparity_frame = np.asarray(disparity_frame)

        frame = {
            "rgb_frame": rgb_frame,
            "disparity_frame": disparity_frame,
            "seg_frame": seg_frame,
            "csv_frame": csv_frame,
        }

        if self.traj_available:
            traj_frame = get_item_between_timestamp(
                self.traj_dat,
                timestamp_ms,
                timestamp_ms + 5000,
                fault_delay=float("inf"),
            )
            frame["traj_frame"] = traj_frame

        for key in csv_frame.keys():
            frame[key] = csv_frame[key]
        return frame


class BengaluruOccupancyDatasetIterator(BengaluruDepthDatasetIterator):
    def __init__(
        self,
        grid_size=(256, 256, 32),  # Occupancy grid size in voxels
        scale=(2.0, 2.0, 0.666),  # voxels per meter
        shift=(0.0, 0.0, 0.0),  # meters
        pc_scale=(500.0, 2500.0, 200.0),
        pc_shift=(100.0, 40.0, 0.0),
        # Minimum number of points in a voxel
        # to be considered occupied
        point_count_threshold=10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        color_2_class = {
            (0, 0, 0): 0,       # Background, black
            (0, 0, 142): 1,     # Vehicle, red
            (220, 20, 60): 2,   # Pedeastrian, blue
        }
        class_2_color = {v: k for k, v in color_2_class.items()}
        num_classes = 3

        self.occupancy_proc = OccupancyProcessor(
            intrinsic_matrix=self.intrinsic_matrix,
            height=self.height,
            width=self.width,
            grid_size=grid_size,
            scale=scale,
            shift=shift,
            pc_scale=pc_scale,
            pc_shift=pc_shift,
            point_count_threshold=point_count_threshold,
            class_2_color=class_2_color,
            color_2_class=color_2_class,
            num_classes=num_classes,
        )

    def __getitem__(self, key):
        frame = super().__getitem__(key)
        return self.occupancy_proc.process_frame(frame)


class OccupancyProcessor:
    def __init__(
        self,
        intrinsic_matrix,
        height,
        width,
        grid_size,  # Occupancy grid size in voxels
        scale,  # voxels per meter
        shift,  # meters
        pc_scale,
        pc_shift,
        # Minimum number of points in a voxel
        # to be considered occupied
        point_count_threshold,
        class_2_color,
        color_2_class,
        num_classes,
    ) -> None:
        super().__init__()

        self.intrinsic_matrix = intrinsic_matrix
        self.height = height
        self.width = width

        self.class_2_color = class_2_color
        self.color_2_class = color_2_class
        self.num_classes = num_classes

        self.pc_scale = pc_scale
        self.pc_shift = pc_shift
        self.point_count_threshold = point_count_threshold
        self.shift = shift
        self.scale = scale
        self.grid_size = grid_size  # Occupancy grid size in unit voxels
        self.occupancy_shape = list(
            map(
                lambda ind: float(self.grid_size[ind] / self.scale[ind]),
                range(len(self.grid_size)),
            )
        )  # Occupancy grid size in unit meters
        self.occupancy_shape = np.array(self.occupancy_shape, dtype=np.float32)

        # print("self.occupancy_shape", self.occupancy_shape)

        self.baseline = 1.0 * 10**-2
        self.fx = self.intrinsic_matrix[0, 0]
        self.fy = self.intrinsic_matrix[1, 1]
        self.cx = self.intrinsic_matrix[0, 2]
        self.cy = self.intrinsic_matrix[1, 2]
        self.focal_length = (self.fx + self.fy) / 2.0

    def transform_points_to_occupancy_grid_vect(
        self, cam_points_orig, semantics
    ):
        occupancy_grid = np.zeros(
            (
                self.grid_size[0],
                self.grid_size[1],
                self.grid_size[2],
                self.num_classes,
            ),
            dtype=np.float32,
        )

        cam_points = cam_points_orig.copy()

        assert (
            cam_points.shape[0] == semantics.shape[0]
        ), "cam_points and \
            semantics must have the same number of points, \
            but got {} and {}".format(
            cam_points.shape[0], semantics.shape[0]
        )

        # Filter out points with inf or nan coordinates
        mask = ~np.isinf(cam_points).any(axis=1) & ~np.isnan(cam_points).any(
            axis=1
        )
        cam_points = cam_points[mask]
        semantics = semantics[mask]

        # Compute grid indices
        ijk = (cam_points / self.occupancy_shape * self.grid_size).astype(int)

        # Filter out points outside the grid
        mask = (
            (0 < ijk[:, 0])
            & (ijk[:, 0] < self.grid_size[0])
            & (0 < ijk[:, 1])
            & (ijk[:, 1] < self.grid_size[1])
            & (0 < ijk[:, 2])
            & (ijk[:, 2] < self.grid_size[2])
        )
        ijk = ijk[mask]
        semantics = semantics[mask]

        # Increment grid cells
        np.add.at(
            occupancy_grid, (ijk[:, 0], ijk[:, 1], ijk[:, 2], semantics), 1
        )

        # Compute occupancy points
        occupancy_points_mask = occupancy_grid >= self.point_count_threshold
        occupancy_points_indices = np.argwhere(occupancy_points_mask)
        occupancy_points = []
        for class_id in range(self.num_classes):
            class_indices = occupancy_points_indices[
                occupancy_points_indices[:, 3] == class_id
            ][:, :3]
            class_points = (
                class_indices / self.grid_size[:3] * self.occupancy_shape[:3]
            ).astype(np.float32)
            class_points = np.concatenate(
                [class_points, np.full((class_points.shape[0], 1), class_id)],
                axis=1,
            )
            occupancy_points.append(class_points)
        occupancy_points = np.concatenate(occupancy_points, axis=0)

        occupancy_grid = occupancy_grid > self.point_count_threshold

        return {
            "occupancy_grid": occupancy_grid,
            "occupancy_points": occupancy_points,
        }

    def transform_points_to_occupancy_grid(self, cam_points_orig, semantics):
        occupancy_grid = np.zeros(
            (
                self.grid_size[0],
                self.grid_size[1],
                self.grid_size[2],
                self.num_classes,
            ),
            dtype=np.float32,
        )

        cam_points = cam_points_orig.copy()

        occupancy_points = set()

        assert (
            cam_points.shape[0] == semantics.shape[0]
        ), "cam_points and \
            semantics must have the same number of points, \
            but got {} and {}".format(
            cam_points.shape[0], semantics.shape[0]
        )

        for index in range(cam_points.shape[0]):
            x, y, z = cam_points[index, :]
            class_id = semantics[index].item()

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
                int(x / self.occupancy_shape[0] * self.grid_size[0]),
                int(y / self.occupancy_shape[1] * self.grid_size[1]),
                int(z / self.occupancy_shape[2] * self.grid_size[2]),
            ]

            # get x,y,z back from i,j,k
            xp = (i / self.grid_size[0]) * self.occupancy_shape[0]
            yp = (j / self.grid_size[1]) * self.occupancy_shape[1]
            zp = (k / self.grid_size[2]) * self.occupancy_shape[2]

            if (
                0 < i < self.grid_size[0]
                and 0 < j < self.grid_size[1]
                and 0 < k < self.grid_size[2]
            ):
                occupancy_grid[i, j, k, class_id] += 1.0
                if (
                    occupancy_grid[i, j, k, class_id]
                    >= self.point_count_threshold
                ):
                    occupancy_points.add((xp, yp, zp, class_id))

        occupancy_points = np.array(list(occupancy_points), dtype=np.float32)

        occupancy_grid = occupancy_grid > self.point_count_threshold

        return {
            "occupancy_grid": occupancy_grid,
            "occupancy_points": occupancy_points,
        }

    def process_frame(self, frame):
        assert isinstance(frame, dict), "frame must be a dict"
        for key in [
            "rgb_frame",
            "disparity_frame",
            "seg_frame",
        ]:
            assert key in frame, "frame must have key {}".format(key)

        disparity = frame["disparity_frame"].astype(np.float32)
        rgb_frame = cv2.cvtColor(frame["rgb_frame"], cv2.COLOR_BGR2RGB)
        seg_frame = cv2.cvtColor(frame["seg_frame"], cv2.COLOR_BGR2RGB)
        seg_frame_class = rgb_seg_to_class(seg_frame, self.color_2_class)

        depth = self.baseline * self.focal_length * np.reciprocal(disparity)
        depth = depth.astype(np.float32)

        hide_mask = np.zeros((self.height, self.width), dtype=bool)
        hide_mask[0 : depth.shape[0] // 2 :,] = True  # noqa
        depth[hide_mask] = float("inf")

        depth[np.isinf(depth)] = 0  # self.baseline * self.focal_length
        depth[np.isnan(depth)] = 0  # self.baseline * self.focal_length

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

        # semantics = seg_frame_class.reshape(
        #     seg_frame_class.shape[0] * seg_frame_class.shape[1],
        #     1
        # )

        semantics = seg_frame_class.flatten()

        points[:, 0] = points[:, 0] * self.pc_scale[0] + self.pc_shift[0]
        points[:, 1] = points[:, 1] * self.pc_scale[1] + self.pc_shift[1]
        points[:, 2] = points[:, 2] * self.pc_scale[2] + self.pc_shift[2]

        correction_angle = np.array((7.0, 0, 0))
        points = rotate_points(points, correction_angle)

        # occupancy_grid_data = self.transform_points_to_occupancy_grid(
        #     points,
        #     semantics
        # )
        occupancy_grid_data = self.transform_points_to_occupancy_grid_vect(
            points, semantics
        )

        occupancy_grid_data["occupancy_points"][:, :3] = rotate_points(
            occupancy_grid_data["occupancy_points"][:, :3], -correction_angle
        )

        # Undo shifting on occupancy_points
        occupancy_grid_data["occupancy_points"][:, 0] = (
            occupancy_grid_data["occupancy_points"][:, 0] - self.pc_shift[0]
        )
        occupancy_grid_data["occupancy_points"][:, 1] = (
            occupancy_grid_data["occupancy_points"][:, 1] - self.pc_shift[1]
        )
        occupancy_grid_data["occupancy_points"][:, 2] = (
            occupancy_grid_data["occupancy_points"][:, 2] - self.pc_shift[2]
        )

        # Undo scaling on occupancy_points
        occupancy_grid_data["occupancy_points"][:, 0] = (
            occupancy_grid_data["occupancy_points"][:, 0] / self.pc_scale[0]
        )
        occupancy_grid_data["occupancy_points"][:, 1] = (
            occupancy_grid_data["occupancy_points"][:, 1] / self.pc_scale[1]
        )
        occupancy_grid_data["occupancy_points"][:, 2] = (
            occupancy_grid_data["occupancy_points"][:, 2] / self.pc_scale[2]
        )

        occupancy_grid_data["occupancy_points"][:, :3] = rotate_points(
            occupancy_grid_data["occupancy_points"][:, :3], correction_angle
        )

        # x, y, z -> x, z, y
        # occupancy_grid_data["occupancy_points"][:, [0, 1, 2]] = (
        #   occupancy_grid_data["occupancy_points"][:, [0, 2, 1]]
        # )

        frame["disparity"] = disparity
        frame["depth"] = depth
        frame["points"] = points
        frame["points_colors"] = points_colors
        frame["occupancy_grid"] = occupancy_grid_data["occupancy_grid"]
        frame["occupancy_points"] = occupancy_grid_data["occupancy_points"]

        return frame


def semantic_pc_to_numpy(semantic_pc):
    """
    semantic_pc: (N, 4)
        (x, y, z, class_id)

    Returns:
        (N * 4)
    """
    assert (
        semantic_pc.shape[1] == 4
    ), "semantic_pc must have 4 columns, but got {}".format(
        semantic_pc.shape[1]
    )
    return semantic_pc.reshape(-1)


def numpy_to_semantic_pc(semantic_pc_numpy):
    """
    semantic_pc_numpy: (N * 4)
        (x, y, z, class_id)

    Returns:
        (N, 4)
    """
    assert (
        semantic_pc_numpy.shape[0] % 4 == 0
    ), "semantic_pc_numpy must have a multiple of 4 rows, but got {}".format(
        semantic_pc_numpy.shape[0]
    )
    return semantic_pc_numpy.reshape(-1, 4)


def semantic_pc_to_colors_and_pc(semantic_pc, class_2_color):
    """
    semantic_pc: (N, 4)
        (x, y, z, class_id)

    Returns:
        points: (N, 3)
        colors: (N, 3)
    """
    points = semantic_pc[:, :3]
    colors = np.array(
        [class_2_color[class_id] for class_id in semantic_pc[:, 3]]
    )

    # substitute black with white

    # colors[
    #     (
    #         colors[:, 0] == 0 &
    #         colors[:, 1] == 0 &
    #         colors[:, 2] == 0
    #     ), :
    # ] = 255

    return points, colors


def rotate_points(points, angles):
    """
    Rotate the set of points by the given euler angles.

    points: numpy array of shape (N, 3)
        The array containing N points with 3D coordinates (x, y, z).
    a, b, c: float
        The euler angles in degrees

    Returns:
    numpy array of shape (N, 3)
        The rotated points.
    """
    a, b, c = angles

    # Convert the angles from degrees to radians
    a = math.radians(a)
    b = math.radians(b)
    c = math.radians(c)

    # Create the rotation matrices
    rotation_matrix_a = np.array(
        [
            [1, 0, 0],
            [0, math.cos(a), -math.sin(a)],
            [0, math.sin(a), math.cos(a)],
        ]
    )
    rotation_matrix_b = np.array(
        [
            [math.cos(b), 0, math.sin(b)],
            [0, 1, 0],
            [-math.sin(b), 0, math.cos(b)],
        ]
    )
    rotation_matrix_c = np.array(
        [
            [math.cos(c), -math.sin(c), 0],
            [math.sin(c), math.cos(c), 0],
            [0, 0, 1],
        ]
    )

    # Rotate the points using the rotation matrices
    rotated_points = np.dot(points, rotation_matrix_a.T)
    rotated_points = np.dot(rotated_points, rotation_matrix_b.T)
    rotated_points = np.dot(rotated_points, rotation_matrix_c.T)

    return rotated_points
