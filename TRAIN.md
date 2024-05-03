# Train

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


python3.9 -m SOccDPT.scripts.train_SOccDPT \
    --version 3 \
    --dataset bdd \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/SOccDPT_V3_dpt_swin2_tiny_256_Aug_11.json


python3.9 -m SOccDPT.scripts.train_SOccDPT \
    --version 3 \
    --dataset bdd \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/SOccDPT_V3_dpt_swin2_tiny_256_Aug_22.json \
    --count 3
```

Train v4 model on BDD Occupancy dataset:
```bash
python3.9 -m SOccDPT.scripts.train_SOccDPT_Occupancy \
    --version 4 \
    --dataset bdd \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/SOccDPT_V4_dpt_swin2_tiny_256_Jul_18.json
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
    --dataset bdd \
    --model_type dpt_swin2_tiny_256 \
    --load checkpoints_pretrained/SOccDPT_V3_dpt_swin2_tiny_256_bdd/liib4duy/checkpoint_epoch_15.pth \
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

python3.9 -m SOccDPT.scripts.eval_others \
    --dataset bdd \
    --model_type zerodepth

python3.9 -m SOccDPT.scripts.eval_others \
    --dataset bdd \
    --model_type packnet
python3.9 -m SOccDPT.scripts.eval_timing
```