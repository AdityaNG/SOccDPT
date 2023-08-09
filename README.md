# SOccDPT

<img src="media/demo.gif" />


<b>Abstract</b> We present SOccDPT, a memory-efficient approach for 3D semantic occupancy prediction from monocular image input using dense predictive transformers. To address the limitations of existing methods trained on structured traffic datasets, we train our model on unstructured datasets including the Indian Driving Dataset and Bengaluru Driving Dataset. Our semi-supervised training pipeline allows SOccDPT to learn from datasets with limited labels by reducing the requirement for manual labelling and substituting it with pseudo-ground truth labels. This broader training enhances our model's ability to handle unstructured traffic scenarios effectively. To overcome memory limitations during training, we introduce patch-wise training where we select a subset of parameters to train each epoch, reducing memory usage during auto-grad graph construction. By considering unstructured traffic and introducing memory-constrained training, SOccDPT achieves a competitive performance as shown by semantic segmentation IoU score of 41.71% and monocular depth estimation RMSE score of 12.4075, even under limited memory constraints and operating at a competitive frequency of 47 Hz. We have made our code and dataset augmentations public.


# Getting Started

## Docker Environment

To build, use:
```bash
DOCKER_BUILDKIT=1 docker compose build
```

To run the interactive shell, use:
```bash
docker compose run dev
```

## Train

Train v1 model on IDD dataset:
```bash
python3.9 -m SOccDPT.scripts.train_SOccDPT \
    --version 1 \
    --dataset idd \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/SOccDPT_V1_dpt_swin2_tiny_256_Jul_18.json
```

Train v1 model on BDD dataset:
```bash
python3.9 -m SOccDPT.scripts.train_SOccDPT \
    --version 1 \
    --dataset bdd \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/SOccDPT_V1_dpt_swin2_tiny_256_Jul_18.json
```

Train v2 model on IDD dataset:
```bash
python3.9 -m SOccDPT.scripts.train_SOccDPT \
    --version 2 \
    --dataset idd \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/SOccDPT_V2_dpt_swin2_tiny_256_Jul_18.json
```

Train v2 model on BDD dataset:
```bash
python3.9 -m SOccDPT.scripts.train_SOccDPT \
    --version 2 \
    --dataset bdd \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/SOccDPT_V2_dpt_swin2_tiny_256_Jul_18.json
```


Train v3 model on IDD dataset:
```bash
python3.9 -m SOccDPT.scripts.train_SOccDPT \
    --version 3 \
    --dataset idd \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/SOccDPT_V3_dpt_swin2_tiny_256_Jul_18.json


python3.9 -m SOccDPT.scripts.train_SOccDPT \
    --version 3 \
    --dataset idd \
    --model_type dpt_swin2_base_384 \
    --count 3 \
    --device cuda:0 \
    --sweep_json config/SOccDPT_V3_dpt_swin2_base_384_Jul_18.json
```

Train v3 model on BDD dataset:
```bash
python3.9 -m SOccDPT.scripts.train_SOccDPT \
    --version 3 \
    --dataset bdd \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/SOccDPT_V3_dpt_swin2_tiny_256_Jul_18.json
```

# Evaluate PatchWise

```bash
python3.9 -m SOccDPT.scripts.eval_patchwise
```

# ONNX

Export
```bash
python3.9 -m SOccDPT.scripts.export_SOccDPT --version 3 \
    --dataset bdd \
    --model_type dpt_swin2_tiny_256 \
    --load checkpoints_pretrained/SOccDPT_V3_dpt_swin2_tiny_256_bdd/qmjmgfu1/checkpoint_epoch_15.pth \
    --export_path onnx/SOccDPT.onnx

python3.9 -m SOccDPT.scripts.export_SOccDPT --version 3 \
    --dataset idd \
    --model_type dpt_swin2_tiny_256 \
    --load checkpoints_pretrained/SOccDPT_V3_dpt_swin2_tiny_256_idd/cq3j88p0/checkpoint_epoch_15.pth \
    --export_path onnx/SOccDPT_idd.onnx
```

Run
```bash
python3.9 -m SOccDPT.scripts.run_SOccDPT_onnx \
    --dataset bdd \
    --model_type dpt_swin2_tiny_256 \
    --load onnx/SOccDPT.onnx

python3.9 -m SOccDPT.scripts.run_SOccDPT_onnx \
    --dataset idd \
    --model_type dpt_swin2_tiny_256 \
    --load onnx/SOccDPT_idd.onnx
```

# Evaluate Models

```bash
python3.9 -m SOccDPT.scripts.eval_SOccDPT \
    --version 3 \
    --dataset bdd \
    --model_type dpt_swin2_tiny_256 \
    --load checkpoints_pretrained/SOccDPT_V3_dpt_swin2_tiny_256_bdd/qmjmgfu1/checkpoint_epoch_15.pth

python3.9 -m SOccDPT.scripts.eval_SOccDPT \
    --version 3 \
    --dataset idd \
    --model_type dpt_swin2_tiny_256 \
    --load checkpoints_pretrained/SOccDPT_V3_dpt_swin2_tiny_256_idd/zyigujaa/checkpoint_epoch_15.pth

python3.9 -m SOccDPT.scripts.eval_others \
    --dataset bdd \
    --model_type DPT_SwinV2_T_256

python3.9 -m SOccDPT.scripts.eval_others \
    --dataset bdd \
    --model_type DPT_Hybrid

python3.9 -m SOccDPT.scripts.eval_others \
    --dataset bdd \
    --model_type DPT_Large

python3.9 -m SOccDPT.scripts.eval_others \
    --dataset bdd \
    --model_type monodepth2

python3.9 -m SOccDPT.scripts.eval_others \
    --dataset bdd \
    --model_type manydepth

python3.9 -m SOccDPT.scripts.eval_timing
```