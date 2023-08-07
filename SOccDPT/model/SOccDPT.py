import os
from typing import Type

import numpy as np
import torch
import torch.nn as nn
import yaml

from .base_model import BaseModel
from .blocks import Interpolate
from .dpt import DPT, DPTDepthModel, DPTSegmentationModel

# from .backbones.vit_3d import ViT3D

from ..datasets.bdd_helper import DEFAULT_CALIB

cpu_device = torch.device("cpu")

DEFAULT_CAMERA_INTRINSICS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

default_depth_models = {
    "dpt_beit_large_512": "weights/dpt_beit_large_512.pt",
    "dpt_beit_large_384": "weights/dpt_beit_large_384.pt",
    "dpt_beit_base_384": "weights/dpt_beit_base_384.pt",
    "dpt_swin2_large_384": "weights/dpt_swin2_large_384.pt",
    "dpt_swin2_base_384": "weights/dpt_swin2_base_384.pt",
    "dpt_swin2_tiny_256": "weights/dpt_swin2_tiny_256.pt",
    "dpt_swin_large_384": "weights/dpt_swin_large_384.pt",
    "dpt_next_vit_large_384": "weights/dpt_next_vit_large_384.pt",
    "dpt_levit_224": "weights/dpt_levit_224.pt",
    "dpt_large_384": "weights/dpt_large_384.pt",
    "dpt_hybrid_384": "weights/dpt_hybrid_384.pt",
}

default_seg_models = {
    "dpt_beit_large_512": None,
    "dpt_beit_large_384": None,
    "dpt_beit_base_384": None,
    "dpt_swin2_large_384": None,
    "dpt_swin2_base_384": None,
    "dpt_swin2_tiny_256": None,
    "dpt_swin_large_384": None,
    "dpt_next_vit_large_384": None,
    "dpt_levit_224": None,
    "dpt_large_384": None,
    "dpt_hybrid_384": None,
}

model_types = default_depth_models.keys()


class SOccDPT(BaseModel):
    def __init__(
        self,
        model_type="dpt_swin2_tiny_256",
        backbone="swin2t16_256",
        path=None,
        num_classes: int = 3,
        camera_intrinsics_yaml=DEFAULT_CALIB,
        depth_scale=1.0,
        depth_shift=0.0,
        point_compute_method="torch",
        **kwargs
    ):
        super(SOccDPT, self).__init__(**kwargs)

        ##########################
        # Load constants
        self.backbone = backbone
        self.model_type = model_type
        self.path = path
        self.scale = depth_scale
        self.shift = depth_shift
        self.num_classes = num_classes

        features = kwargs["features"] if "features" in kwargs else 256
        self.features = features

        assert point_compute_method in ("torch", "numpy")
        self.point_compute_method = point_compute_method
        ##########################
        # Loading camera intrinsics
        self.camera_intrinsics_yaml = os.path.expanduser(
            camera_intrinsics_yaml
        )
        with open(self.camera_intrinsics_yaml, "r") as stream:
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

        self.fx = self.intrinsic_matrix[0, 0]
        self.fy = self.intrinsic_matrix[1, 1]
        self.cx = self.intrinsic_matrix[0, 2]
        self.cy = self.intrinsic_matrix[1, 2]

        self.width = self.cam_settings["Camera.width"]
        self.height = self.cam_settings["Camera.height"]
        ##########################
        # ViT 3D
        # volume_size = (256, 32, 256)
        # volume_size = (128, 16, 126)
        # volume_patch_size = (4, 4, 4)
        # frames = 1
        # frame_patch_size = 1
        # num_classes = self.num_classes
        # dim = max(self.num_classes // 2, 1)
        # depth = 3
        # heads = 3
        # mlp_dim = dim
        # pool = 'cls'
        # channels = 3
        # dim_head = 64
        # dropout = 0.
        # emb_dropout = 0.
        # self.vid_3d = ViT3D(
        #     volume_size = volume_size,
        #     volume_patch_size = volume_patch_size,
        #     frames = frames,
        #     frame_patch_size = frame_patch_size,
        #     num_classes = num_classes,
        #     dim = dim,
        #     depth = depth,
        #     heads = heads,
        #     mlp_dim = mlp_dim,
        #     pool = pool,
        #     channels = channels,
        #     dim_head = dim_head,
        #     dropout = dropout,
        #     emb_dropout = emb_dropout,
        # )

        # self.grid_height, self.grid_width, self.grid_depth = volume_size
        # self.x_min, self.x_step = 0.0, 1.0
        # self.y_min, self.y_step = 0.0, 1.0
        # self.z_min, self.z_step = 0.0, 1.0
        ##########################

    def forward(self, x: torch.Tensor):
        """
        x: (batch_size, 3, H, W)
        (H, W) are determined by architecture,
        look at SOccDPT.loader.load_transforms for mode details
        """
        assert (
            False
        ), "Not implemented, take input batch and produce inv_depth, \
            segmentation and call \
            self.get_semantic_occupancy(inv_depth, segmentation)"

        # inv_depth = self.depth_net(depth_input)
        # segmentation = self.seg_net(seg_input)
        # return self.get_semantic_occupancy(inv_depth, segmentation)

    def get_semantic_occupancy(self, inv_depth, segmentation):
        device = self.get_device()

        if len(inv_depth.shape) == 3:
            inv_depth = inv_depth.unsqueeze(1)

        inv_depth = torch.nn.functional.interpolate(
            # inv_depth.unsqueeze(1),
            inv_depth,
            size=(self.height, self.width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        segmentation = torch.nn.functional.interpolate(
            segmentation,
            size=(self.height, self.width),
            mode="nearest",  # nearest-exact
        ).squeeze()

        if len(inv_depth.shape) == 2:
            inv_depth = inv_depth.unsqueeze(0)

        depth = self.scale * inv_depth + self.shift
        depth[depth < 1e-8] = 1e-8
        depth = 1.0 / depth

        depth[torch.isinf(depth)] = float("inf")
        depth[torch.isnan(depth)] = float("inf")

        if self.point_compute_method == "torch":
            ############################
            # Parallel batch operation
            # Using torch takes up a lot of GPU memory

            # Create a grid of U and V values for all pixels in the image
            U, V = torch.meshgrid(
                torch.arange(self.height, device=device, dtype=torch.float32),
                torch.arange(self.width, device=device, dtype=torch.float32),
            )

            # Repeat the U and V grids for each batch
            U = U.unsqueeze(0).repeat(inv_depth.shape[0], 1, 1)
            V = V.unsqueeze(0).repeat(inv_depth.shape[0], 1, 1)

            # Compute X, Y, and Z values using broadcasting
            X = (V - self.cx) * depth / self.fx
            Y = (U - self.cy) * depth / self.fy
            Z = depth

            # Combine X, Y, and Z values into a single tensor
            points_batched = torch.stack([X, Y, Z], dim=3)
            ############################
        else:
            points_batched = []
            for batch_index in range(inv_depth.shape[0]):
                points = np.zeros(
                    (self.height, self.width, 3),
                    dtype=np.float32,
                )

                U, V = np.ix_(
                    np.arange(self.height), np.arange(self.width)
                )  # pylint: disable=unbalanced-tuple-unpacking
                Z = depth[batch_index, :].detach().cpu().numpy()

                X = (V - self.cx) * Z / self.fx
                Y = (U - self.cy) * Z / self.fy

                points[:, :, 0] = X
                points[:, :, 1] = Y
                points[:, :, 2] = Z

                points_batched.append(points)

            points_batched = np.array(points_batched, dtype=np.float32)
            points_batched = torch.from_numpy(points_batched)

        # semantics_3D = segmentation.reshape(
        #     -1, self.num_classes, self.height * self.width
        # )
        # occupancy_grid = self.points_to_occupancy_grid(points, semantics_3D)
        # occupancy_grid = self.vid_3d(occupancy_grid)

        return inv_depth, segmentation, points_batched

    def points_to_occupancy_grid(self, points, semantics_3D):
        """
        points: (N, P, 3)
        semantics_3D: (N, P, self.num_classes)

        N is batch size
        P is number of points
        points[N, P, :] is the 3D point which corresponds to the
        pixel semantics_3D[N, P, :]

        returns:
            occupancy_grid: (
                N,
                self.grid_height,
                self.grid_width,
                self.grid_depth,
                self.num_classes
            )
        """
        device = self.get_device()

        occupancy_grid = torch.zeros(
            (
                semantics_3D.shape[0],
                self.grid_height,
                self.grid_width,
                self.grid_depth,
                self.num_classes,
            ),
            dtype=torch.float32,
            device=device,
        )

        # Poplulate occupancy grid efficiently using torch
        x = points[:, :, 0]
        y = points[:, :, 1]
        z = points[:, :, 2]

        x = (x - self.x_min) / self.x_step
        y = (y - self.y_min) / self.y_step
        z = (z - self.z_min) / self.z_step

        x = torch.clamp(x, 0, self.grid_width - 1)
        y = torch.clamp(y, 0, self.grid_height - 1)
        z = torch.clamp(z, 0, self.grid_depth - 1)

        # Round to nearest integer
        x = torch.round(x).astype(torch.int64)
        y = torch.round(y).astype(torch.int64)
        z = torch.round(z).astype(torch.int64)

        for i in range(semantics_3D.shape[0]):
            occupancy_grid[i, y[i, :], x[i, :], z[i, :], :] = semantics_3D[
                i, :, :
            ]

        return occupancy_grid


DEPTH_l39icv3q = "checkpoints_pretrained/depth_dpt_hybrid/l39icv3q/checkpoint_epoch15.pth"  # noqa: E501
SEG_wrlnq5jb = "checkpoints_pretrained/seg_dpt_hybrid/wrlnq5jb/checkpoint_epoch15.pth"  # noqa: E501


class SOccDPT_V1(SOccDPT):
    def __init__(
        self,
        load_depth: str = DEPTH_l39icv3q,
        load_seg: str = SEG_wrlnq5jb,
        **kwargs
    ):
        super(SOccDPT_V1, self).__init__(**kwargs)

        from .loader import load_model

        ##########################
        # Loading depth network
        depth_model_weights = load_depth
        if depth_model_weights is None:
            depth_model_weights = default_depth_models[self.model_type]

        self.depth_net = load_model(
            DPTDepthModel,
            dict(
                non_negative=True,
            ),
            cpu_device,
            depth_model_weights,
            self.model_type,
        )
        self.pretrained = self.depth_net.pretrained
        ##########################

        ##########################
        # Loading seg network
        seg_model_weights = load_seg
        if seg_model_weights is None:
            seg_model_weights = default_seg_models[self.model_type]
        self.seg_net = load_model(
            DPTSegmentationModel,
            dict(
                num_classes=self.num_classes,
            ),
            cpu_device,
            seg_model_weights,
            self.model_type,
        )
        ##########################

        ##########################
        # Load model weights
        self.load_net(self.path)

    def forward(self, x: torch.Tensor):
        inv_depth = self.depth_net(x)
        segmentation = self.seg_net(x)

        return self.get_semantic_occupancy(inv_depth, segmentation)


class SOccDPT_V2(SOccDPT):
    def __init__(self, **kwargs):
        super(SOccDPT_V2, self).__init__(**kwargs)

        from .loader import load_model

        ##########################
        # Loading DPT backbone
        identity_head = nn.Sequential(
            nn.Identity(),
        )

        self.pretrained = load_model(
            DPT,
            dict(
                head=identity_head,
            ),  # TODO: Should I add the other kwargs?
            cpu_device,
            None,
            self.model_type,
        )
        ##########################

        ##########################
        # Loading depth and seg heads

        self.features = kwargs["features"] if "features" in kwargs else 256
        head_features_1 = (
            kwargs["head_features_1"]
            if "head_features_1" in kwargs
            else self.features
        )
        head_features_2 = (
            kwargs["head_features_2"] if "head_features_2" in kwargs else 32
        )
        non_negative = (
            kwargs["non_negative"] if "non_negative" in kwargs else True
        )

        kwargs.pop("head_features_1", None)
        kwargs.pop("head_features_2", None)
        kwargs.pop("non_negative", None)

        self.depth_head = nn.Sequential(
            nn.Conv2d(
                head_features_1,
                head_features_1 // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                head_features_1 // 2,
                head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(
                self.features,
                self.features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(self.features, self.num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )
        ##########################

        ##########################
        # Load model weights
        self.load_net(self.path)

    def forward(self, x: torch.Tensor):
        backbone_features = self.pretrained.forward(x)

        inv_depth = self.depth_head(backbone_features)
        segmentation = self.seg_head(backbone_features)

        return self.get_semantic_occupancy(inv_depth, segmentation)


class SOccDPT_V3(SOccDPT):
    def __init__(self, load_depth: str = DEPTH_l39icv3q, **kwargs):
        super(SOccDPT_V3, self).__init__(**kwargs)

        from .loader import load_model

        ##########################
        # Loading depth network
        depth_model_weights = load_depth
        if depth_model_weights is None:
            depth_model_weights = default_depth_models[self.model_type]

        print("Loading depth net")
        self.depth_net = load_model(
            DPTDepthModel,
            dict(
                non_negative=True,
                return_features=True,
            ),
            cpu_device,
            depth_model_weights,
            self.model_type,
        )
        self.depth_net.return_features = True
        self.pretrained = self.depth_net.pretrained
        ##########################

        ##########################
        # Seg head
        self.seg_head = nn.Sequential(
            nn.Conv2d(
                self.features,
                self.features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(self.features, self.num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )
        ##########################

        ##########################
        # Load model weights
        self.load_net(self.path)

    def forward(self, x: torch.Tensor):
        inv_depth, backbone_features = self.depth_net.forward(x)
        segmentation = self.seg_head(backbone_features)

        return self.get_semantic_occupancy(inv_depth, segmentation)


SOccDPT_versions = {
    1: SOccDPT_V1,
    2: SOccDPT_V2,
    3: SOccDPT_V3,
}


# Use the following classes to convert the SOccDPT model into a depth
# or segmentation model
class DepthNet:
    def __init__(self, net: Type[SOccDPT]) -> None:
        self.net = net

    def __call__(self, x: torch.Tensor):
        y_disp_pred, _, _ = self.net(x)
        return y_disp_pred

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()


class SegNet:
    def __init__(self, net: Type[SOccDPT]) -> None:
        self.net = net

    def __call__(self, x: torch.Tensor):
        _, y_seg_pred, _ = self.net(x)
        return y_seg_pred

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()
