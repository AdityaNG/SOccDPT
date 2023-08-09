import numpy as np
import torch
import torch.hub


@torch.no_grad()
def eval_net(net: torch.nn.Module, input_tensor: torch.tensor, N: int = 50):
    net.eval()

    # for _ in tqdm(range(N)):
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()

    for _ in range(N):
        _ = net(input_tensor)

    end_time.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_time.elapsed_time(end_time)

    fps = 1000.0 / elapsed_time_ms  # Frames per second

    # Count the number of parameters
    # num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in net.parameters())

    return fps, num_params


def eval_midas(device):
    DEPTH_NETS = [
        "DPTDepthModel",
        "DPT_Hybrid",
        "DPT_Large",
        "DPT_LeViT_224",
        "DPT_SwinV2_B_384",
        "DPT_SwinV2_L_384",
        "DPT_SwinV2_T_256",
        "DPT_Swin_L_384",
        "MiDaS",
        "MiDaS_small",
        "MidasNet",
        "MidasNet_small",
    ]
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    # print(dir(midas_transforms))
    # exit()

    for depth_net_type in DEPTH_NETS:
        if depth_net_type == "DPT_Large" or depth_net_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        elif "LeViT" in depth_net_type:
            transform = midas_transforms.levit_transform
        elif "DPT_SwinV2_T_256" == depth_net_type:
            transform = midas_transforms.swin256_transform
        elif "DPT_Swin_L_384" == depth_net_type:
            transform = midas_transforms.swin384_transform
        elif "DPT_BEiT_L_512" == depth_net_type:
            transform = midas_transforms.beit512_transform
        elif (
            "DPT_BEiT_B_384" == depth_net_type
            or "DPT_BEiT_L_384" == depth_net_type
        ):
            transform = midas_transforms.swin384_transform
        elif (
            "DPT_SwinV2_B_384" == depth_net_type
            or "DPT_SwinV2_L_384" == depth_net_type
        ):
            transform = midas_transforms.swin384_transform
        elif "MiDaS" == depth_net_type:
            transform = midas_transforms.default_transform
        else:
            transform = midas_transforms.small_transform

        net = torch.hub.load("intel-isl/MiDaS", depth_net_type)
        net = net.to(device=device)

        # Perform a dummy forward pass to calculate FPS
        img = np.zeros((1920, 1080, 3), np.uint8)
        input_tensor = transform(img).to(device=device)
        fps, num_params = eval_net(net, input_tensor)

        print(f"Model: {depth_net_type}")
        print(f"FPS: {fps:.4f}")
        print(f"Number of Parameters: {num_params}")
        print()


def eval_SOccDPT(device):
    from ..model.SOccDPT import SOccDPT_versions
    from ..model.loader import load_model, load_transforms

    # for v in SOccDPT_versions:
    for v in SOccDPT_versions:
        SOccDPT = SOccDPT_versions[v]
        # model_type="dpt_swin2_tiny_256"
        for model_type in [
            # "dpt_swin2_large_384",
            # "dpt_swin2_base_384",
            "dpt_swin2_tiny_256",
            # "dpt_swin_large_384",
            # "dpt_large_384",
            # "dpt_hybrid_384",
        ]:
            net = load_model(
                arch=SOccDPT,
                model_kwargs=dict(),
                model_path=None,
                device=device,
                model_type=model_type,
            )
            transform, net_w, net_h = load_transforms(
                model_type=model_type,
            )

            # print('model_type', model_type)
            # print('net_w, net_h', net_w, net_h)

            # Perform a dummy forward pass to calculate FPS
            img = np.zeros((1920, 1080, 3), np.uint8)
            input_tensor = transform({"image": img})["image"]
            input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
            input_tensor = input_tensor.to(device=device)

            # print('input_tensor.shape', input_tensor.shape)

            fps, num_params = eval_net(net, input_tensor)

            print(f"Model: SOccDPT_V{str(v)}_{model_type}")
            print(f"FPS: {fps:.4f}")
            print(f"Number of Parameters: {num_params}")
            print()

            del net
            torch.cuda.empty_cache()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_midas(device)
    # eval_SOccDPT(device)
