# SOccDPT

<img src="media/demo.gif" />

[Project Page](https://adityang.github.io/SOccDPT)

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

Train v3 model on BDD dataset:
```bash
python3.9 -m SOccDPT.scripts.train_SOccDPT \
    --version 3 \
    --dataset bdd \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/SOccDPT_V3_dpt_swin2_tiny_256_Aug_22.json \
```

Read more in [TRAIN.md](TRAIN.md)

## Cite

Cite our work if you find it useful

```bibtex
@article{nalgunda2024soccdpt,
  author = {Nalgunda, Aditya Ganesh},
  title = {SOccDPT: 3D Semantic Occupancy from Dense Prediction Transformers trained under memory constraints},
  journal = {Advances in Artificial Intelligence and Machine Learning},
  volume = {4},
  number = {2},
  pages = {2201--2212},
  year = {2024}
}
```
