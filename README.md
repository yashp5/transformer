# Transformer Training and Inference Optimization

This project explores optimization techniques for training and inference of transformer models, specifically focusing on distributed training scaling efficiency and model quantization.

## Project Overview

The project implements and analyzes:
- Distributed training with PyTorch DDP (DistributedDataParallel)
- Strong scaling analysis across multiple GPUs
- Model quantization for inference optimization

## Project Milestones

✅ Distributed Training Implementation
- Implemented PyTorch DDP for multi-GPU training
- Added detailed timing breakdowns for training phases
- Implemented data parallelism with proper gradient synchronization

✅ Scaling Analysis
- Tested with 1, 2, and 4 GPUs configurations
- Analyzed scaling efficiency with different batch sizes (64, 128)
- Measured compute vs. communication overhead

✅ Model Optimization
- Dynamic quantization implementation
- Performance comparison between base and quantized models
- Training and inference metrics collection

## Repository Structure

```
gpt/
├── README.md
├── gpt.py                  # Base GPT model implementation
├── train.py               # Single-GPU training script
├── train_dist.py          # Distributed training implementation
├── eval.py                # Evaluation and quantization script
└── slurm/                 # SLURM job submission scripts
    ├── train_b64_g1.slurm
    ├── train_b64_g2.slurm
    ├── train_b64_g4.slurm
    ├── train_b128_g1.slurm
    ├── train_b128_g2.slurm
    └── train_b128_g4.slurm
```

## Usage

### Single GPU Training
```bash
python train.py --batch_size 64 --epochs 1 --learning_rate 1e-4
```

### Distributed Training
```bash
python train_dist.py --num_gpus 4 --batch_size 128 --epochs 1
```

### Model Evaluation and Quantization
```bash
python eval.py
```

## Results

### Distributed Training Scaling Analysis

| GPUs | Batch-size 64 |         | Batch-size 128 |         |
|------|---------------|---------|----------------|---------|
|      | Time(sec)     | Speedup | Time(sec)      | Speedup |
| 1    | 2233.50       | 1.00    | 2082.29        | 1.00    |
| 2    | 1095.70       | 2.04    | 941.42         | 2.21    |
| 4    | 824.49        | 2.71    | 489.62         | 4.25    |

### Communication vs Computation Analysis

| GPUs | Batch-size 64 |           | Batch-size 128 |           |
|------|---------------|-----------|----------------|-----------|
|      | Compute(sec)  | Comm(sec) | Compute(sec)   | Comm(sec) |
| 2    | 1047.59       | 40.02     | 907.32         | 25.67     |
| 4    | 776.61        | 41.38     | 463.90         | 21.22     |

## Key Observations

1. **Scaling Efficiency**
   - Near-linear scaling (2.04x) with 2 GPUs for batch size 64
   - Super-linear scaling (4.25x) with 4 GPUs for batch size 128
   - Larger batch sizes show better scaling efficiency

2. **Communication Overhead**
   - Communication overhead decreases with larger batch sizes
   - Batch size 128 shows better compute-to-communication ratio
   - For batch size 128:
     - 2 GPUs: ~35x more compute than communication time
     - 4 GPUs: ~22x more compute than communication time

3. **Strong Scaling Characteristics**
   - Shows typical strong scaling behavior with diminishing returns
   - Larger batch sizes help amortize communication overhead
   - Sub-linear speedup due to communication overhead and Amdahl's Law

## Future Work

- [ ] Implement weak scaling analysis
- [ ] Explore mixed-precision training
- [ ] Add pipeline parallelism
- [ ] Implement more advanced quantization techniques
- [ ] Benchmark on different GPU architectures

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
