# SOccDPT

<b>Abstract</b> We present SOccDPT, a memory-efficient approach for 3D semantic occupancy dense prediction using transformers. Our method generates a 3D semantic occupancy point set from a monocular image input. We explore multiple architectural variations, including SOccDPT V1 to V5, capturing different aspects of the input data for accurate predictions. To overcome memory limitations, we introduce patchwise training, allowing training on systems with only 8 GB of video RAM. We randomly select a subset of parameters to train each epoch, reducing memory usage during autograd graph construction. Addressing the limitations of existing methods trained on structured traffic datasets, we train our model on unstructured datasets like Bengaluru Depth Dataset and Indian Driving Dataset. This broader training enhances our model's ability to handle unstructured traffic scenarios effectively. Evaluation on Bengaluru Depth Dataset, KITTI, and Indian Driving Dataset demonstrates the effectiveness of SOcc DPT. Our models achieve competitive performance, even under limited memory constraints. By introducing memory-constrained training and considering unstructured traffic, our work contributes to the field of computer vision and opens new possibilities for vision transformers in challenging environments with limited computational resources.


# Getting Started

## Docker Environment

To build, use:
```bash
docker compose build
```

To run the interactive shell, use:
```bash
docker compose run dev
```

## Train

```bash
python3 -m SOccDPT.scripts.train_SOccDPT
```