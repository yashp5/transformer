# GPT

### training_b64_g1

2024-12-10 23:07:29,977 - Epoch 0: train loss 0.6340, val loss 1.6000

2024-12-10 23:07:02,646 -
Epoch 0 timing breakdown:
Total epoch time: 2233.50 seconds
Data loading time: 11.67 seconds
Data movement time: 2.79 seconds
Training compute time: 2168.11 seconds
Training communication time: 50.93 seconds

### training_b128_g1

2024-12-10 23:05:20,618 - Epoch 0: train loss 0.8363, val loss 1.4769

2024-12-10 23:04:28,806 -
Epoch 0 timing breakdown:
Total epoch time: 2082.29 seconds
Data loading time: 9.78 seconds
Data movement time: 2.05 seconds
Training compute time: 2045.15 seconds
Training communication time: 25.31 seconds

### training_b64_g2

```bash
2024-12-11 16:14:45,133 - Average metrics:
    Total epoch time: 1095.70 seconds
    Data loading time: 6.28 seconds
    Data movement time: 1.82 seconds
    Training compute time: 1047.59 seconds
    Training communication time: 40.02 seconds
    Train loss: 0.6123
    Val loss: 1.7059

2024-12-11 16:14:45,135 - Rank 0 -
    Rank 0 Epoch 0 timing breakdown:
        Average Loss: 1.22
        Total epoch time: 1384.64 seconds
        Data loading time: 6.37 seconds
        Data movement time: 1.80 seconds
        Training compute time: 1335.39 seconds
        Training communication time: 41.09 seconds

2024-12-11 16:14:45,133 - Rank 1 -
    Rank 1 Epoch 0 timing breakdown:
        Average Loss: 1.22
        Total epoch time: 806.75 seconds
        Data loading time: 6.18 seconds
        Data movement time: 1.83 seconds
        Training compute time: 759.79 seconds
        Training communication time: 38.94 seconds

2024-12-11 16:15:00,831 - Rank 0 - Epoch 0: train loss 1.22 , val loss 1.5059
```

### training_b128_g2

Average metrics:
Total epoch time: 941.42 seconds
Data loading time: 6.81 seconds
Data movement time: 1.62 seconds
Training compute time: 907.32 seconds
Training communication time: 25.67 seconds
Train loss: 0.8419
Val loss: 1.5353

2024-12-11 16:46:49,488 - Rank 0 - Epoch 0: train loss 0.8419, val loss 1.5353

2024-12-11 16:46:21,186 - Rank 1 -
Rank 1 Epoch 0 timing breakdown:
Average Loss: 1.36
Total epoch time: 693.97 seconds
Data loading time: 7.60 seconds
Data movement time: 1.78 seconds
Training compute time: 660.44 seconds
Training communication time: 24.15 seconds

2024-12-11 16:46:21,189 - Rank 0 -
Rank 0 Epoch 0 timing breakdown:
Average Loss: 1.36
Total epoch time: 1188.86 seconds
Data loading time: 6.01 seconds
Data movement time: 1.45 seconds
Training compute time: 1154.20 seconds
Training communication time: 27.19 seconds

### training_b64_g4

Average metrics:
Total epoch time: 824.49 seconds
Data loading time: 4.92 seconds
Data movement time: 1.58 seconds
Training compute time: 776.61 seconds
Training communication time: 41.38 seconds
Train loss: 0.6063
Val loss: 1.7253

2024-12-11 15:02:53,761 - Rank 0 - Epoch 0: train loss 0.6063, val loss 1.7253

2024-12-11 15:02:44,946 - Rank 0 -
Rank 0 Epoch 0 timing breakdown:
Average Loss: 1.21
Total epoch time: 833.37 seconds
Data loading time: 5.03 seconds
Data movement time: 1.56 seconds
Training compute time: 784.32 seconds
Training communication time: 42.47 seconds

2024-12-11 15:02:44,946 - Rank 3 -
Rank 3 Epoch 0 timing breakdown:
Average Loss: 1.21
Total epoch time: 816.87 seconds
Data loading time: 4.83 seconds
Data movement time: 1.58 seconds
Training compute time: 770.15 seconds
Training communication time: 40.31 seconds

2024-12-11 15:02:44,947 - Rank 2 -
Rank 2 Epoch 0 timing breakdown:
Average Loss: 1.21
Total epoch time: 827.95 seconds
Data loading time: 4.93 seconds
Data movement time: 1.60 seconds
Training compute time: 780.41 seconds
Training communication time: 41.01 seconds

2024-12-11 15:02:44,947 - Rank 1 -
Rank 1 Epoch 0 timing breakdown:
Average Loss: 1.21
Total epoch time: 819.76 seconds
Data loading time: 4.90 seconds
Data movement time: 1.59 seconds
Training compute time: 771.56 seconds
Training communication time: 41.72 seconds

### training_b128_g4

Average metrics:
Total epoch time: 489.62 seconds
Data loading time: 3.53 seconds
Data movement time: 0.98 seconds
Training compute time: 463.90 seconds
Training communication time: 21.22 seconds
Train loss: 0.8129
Val loss: 1.5556

2024-12-11 15:00:12,837 - Rank 0 -
Rank 0 Epoch 0 timing breakdown:
Average Loss: 1.34
Total epoch time: 706.41 seconds
Data loading time: 3.58 seconds
Data movement time: 0.96 seconds
Training compute time: 679.58 seconds
Training communication time: 22.29 seconds

2024-12-11 15:00:12,835 - Rank 1 -
Rank 1 Epoch 0 timing breakdown:
Average Loss: 1.34
Total epoch time: 418.27 seconds
Data loading time: 3.54 seconds
Data movement time: 1.00 seconds
Training compute time: 392.74 seconds
Training communication time: 21.00 seconds

2024-12-11 15:00:12,835 - Rank 2 -
Rank 2 Epoch 0 timing breakdown:
Average Loss: 1.34
Total epoch time: 415.71 seconds
Data loading time: 3.48 seconds
Data movement time: 0.97 seconds
Training compute time: 390.45 seconds
Training communication time: 20.81 seconds

2024-12-11 15:00:12,835 - Rank 3 -
Rank 3 Epoch 0 timing breakdown:
Average Loss: 1.34
Total epoch time: 418.10 seconds
Data loading time: 3.51 seconds
Data movement time: 0.99 seconds
Training compute time: 392.81 seconds
Training communication time: 20.79 seconds

2024-12-11 15:00:28,525 - Rank 0 - Epoch 0: train loss 0.8129, val loss 1.5556

## Observations

|       | Batch-size 64 |         | Batch-size 128 |         |
| ----- | ------------- | ------- | -------------- | ------- |
|       | Time(sec)     | Speedup | Time(sec)      | Speedup |
| 1-GPU | 2233.50       | 1.00    | 2082.29        | 1.00    |
| 2-GPU | 1095.70       | 2.04    | 941.42         | 2.21    |
| 4-GPU | 824.49        | 2.71    | 489.62         | 4.25    |

|       | Batch-size 64 |           | Batch-size 128 |           |
| ----- | ------------- | --------- | -------------- | --------- |
|       | Compute(sec)  | Comm(sec) | Compute(sec)   | Comm(sec) |
| 2-GPU | 1047.59       | 40.02     | 907.32         | 25.67     |
| 4-GPU | 776.61        | 41.38     | 463.90         | 21.22     |

We can observe:

- For batch size 64, scaling from 1 to 2 GPUs gives nearly linear speedup (2.04x)
- For batch size 128, we see even better scaling, especially with 4 GPUs (4.25x speedup)
- Overall, larger batch size (128) shows better scaling efficiency than smaller batch size (64)

This is strong scaling because:

1. We're keeping the total problem size (dataset size) constant while increasing the number of GPUs
2. We're measuring how the same total workload speeds up as we add more processors/GPUs
3. The batch sizes per experiment (64 and 128) remain constant across different GPU counts

In contrast, weak scaling would:

- Increase the problem size proportionally with the number of processors
- Keep the work per processor constant
- For ML training, this often means increasing the batch size proportionally with GPU count (e.g., if base batch size is 64, 2 GPUs would use 128, 4 GPUs would use 256)

Our results show typical strong scaling behavior:

- Diminishing returns as we add more GPUs (speedup < number of GPUs)
- Sub-linear speedup (e.g., 4 GPUs gives 2.71x and 4.25x speedup rather than ideal 4x)
- This is expected due to communication overhead and sequential portions of the program (Amdahl's Law)

The larger batch size (128) shows better strong scaling efficiency, likely because the increased computation per GPU helps amortize the communication overhead.

Observations:

- Communication overhead (Comm time) decreases with larger batch size (from ~40s to ~21-25s)
- The ratio of compute to communication time improves with larger batch size
- For batch size 64:
  - 2-GPU: ~26x more compute than comm time
  - 4-GPU: ~19x more compute than comm time
- For batch size 128:
  - 2-GPU: ~35x more compute than comm time
  - 4-GPU: ~22x more compute than comm time
