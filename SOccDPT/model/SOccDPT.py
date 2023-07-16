import os
from typing import Type

import numpy as np
import torch
import torch.nn as nn
import yaml

from ..utils import freeze_pretrained_encoder
from .base_model import BaseModel
from .blocks import Interpolate
from .dpt import DPT, DPTDepthModel, DPTSegmentationModel
from .loader import load_model

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
        path=None,
        num_classes: int = 3,
        camera_intrinsics_yaml=DEFAULT_CALIB,
        depth_scale=1.0,
        depth_shift=0.0,
        **kwargs
    ):
        super(SOccDPT, self).__init__(**kwargs)

        ##########################
        # Load constants
        self.model_type = model_type
        self.path = path
        self.scale = depth_scale
        self.shift = depth_shift
        self.num_classes = num_classes
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

        inv_depth = torch.nn.functional.interpolate(
            inv_depth.unsqueeze(1),
            size=(self.height, self.width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        segmentation = torch.nn.functional.interpolate(
            segmentation,
            size=(self.height, self.width),
            mode="closest-exact",
        ).squeeze()

        depth = self.scale * inv_depth + self.shift
        depth[depth < 1e-8] = 1e-8
        depth = 1.0 / depth

        depth[torch.isinf(depth)] = float("inf")
        depth[torch.isnan(depth)] = float("inf")

        points = torch.zeros(
            (self.height, self.width, 3), dtype=torch.float32, device=device
        )

        U, V = np.ix_(
            np.arange(self.height), np.arange(self.width)
        )  # pylint: disable=unbalanced-tuple-unpacking
        U = torch.from_numpy(U).to(device=device)
        V = torch.from_numpy(V).to(device=device)
        Z = depth

        X = (V - self.cx) * Z / self.fx
        Y = (U - self.cy) * Z / self.fy

        points[:, :, 0] = X
        points[:, :, 1] = Y
        points[:, :, 2] = Z

        return inv_depth, segmentation, points


DEPTH_l39icv3q = "checkpoints/depth_dpt_hybrid/l39icv3q/checkpoint_epoch15.pth"
SEG_wrlnq5jb = "checkpoints/seg_dpt_hybrid/wrlnq5jb/checkpoint_epoch15.pth"


class SOccDPT_V1(SOccDPT):
    def __init__(
        self,
        load_depth: str = DEPTH_l39icv3q,
        load_seg: str = SEG_wrlnq5jb,
        **kwargs
    ):
        super(SOccDPT_V1, self).__init__(**kwargs)

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
        freeze_pretrained_encoder(self.depth_net)
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
        # change_number_of_classes(self.seg_net, self.num_classes)
        freeze_pretrained_encoder(self.seg_net)
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
        super(SOccDPT_V1, self).__init__(**kwargs)

        ##########################
        # Loading DPT backbone
        identity_head = nn.Sequential(
            nn.Identity(),
        )

        self.dpt = load_model(
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
        backbone_features = self.dpt.forward(x)

        inv_depth = self.depth_net(backbone_features)
        segmentation = self.seg_net(backbone_features)

        return self.get_semantic_occupancy(inv_depth, segmentation)


class SOccDPT_V3(SOccDPT):
    def __init__(self, load_depth: str = DEPTH_l39icv3q, **kwargs):
        super(SOccDPT_V3, self).__init__(**kwargs)

        ##########################
        # Loading depth network
        depth_model_weights = load_depth
        if depth_model_weights is None:
            depth_model_weights = default_depth_models[self.model_type]

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
        # self.depth_net.return_features = True
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
