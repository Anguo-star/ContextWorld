# ContextWorld Test adaptation metrics

These tables report Test accuracy, history benefit, and response error for 136 runs in 86 comparable task, model, and training configurations. Values are mean ± sample standard deviation across training seeds; n gives the number of seeds. A single seed has no standard deviation.

Accuracy and History are percentages (higher is better). History is the proportion of cases where matching history yields lower prediction error than alternative history. Speed Accuracy and History use the same unseen interpolation gate score. NRE is normalized latent response error (lower is better).

Only measured configurations are listed. DINO-WM continuation is labeled Original → Mixed. Multi-step prediction results are reported separately on Development.

## Speed

| Model | Training | n | Accuracy (%) ↑ | History (%) ↑ | NRE ↓ |
|---|---|---:|---:|---:|---:|
| LeWM | Original | 3 | 58.04 ± 1.70 | 58.04 ± 1.70 | 0.896 ± 0.027 |
| LeWM | Mixed | 3 | 98.19 ± 0.57 | 98.19 ± 0.57 | 0.042 ± 0.011 |
| LeWM | Original → Mixed Joint | 1 | 96.89 | 96.89 | 0.055 |
| LeWM | Original → Mixed Frozen Encoder | 1 | 98.44 | 98.44 | 0.030 |
| PLDM | Original | 3 | 48.41 ± 4.33 | 48.41 ± 4.33 | 0.922 ± 0.026 |
| PLDM | Mixed | 3 | 96.78 ± 2.02 | 96.78 ± 2.02 | 0.084 ± 0.013 |
| PLDM | Original → Mixed Joint | 1 | 98.33 | 98.33 | 0.047 |
| PLDM | Original → Mixed Frozen Encoder | 1 | 96.44 | 96.44 | 0.062 |
| DINO-WM | Original | 1 | 57.11 | 57.11 | 0.919 |
| DINO-WM | Mixed | 3 | 59.41 ± 3.87 | 59.41 ± 3.87 | 0.961 ± 0.011 |
| DINO-WM | Original → Mixed | 1 | 84.78 | 84.78 | 0.411 |

## Pusher displacement

| Model | Training | n | Accuracy (%) ↑ | History (%) ↑ | NRE ↓ |
|---|---|---:|---:|---:|---:|
| LeWM | Original | 3 | 48.96 ± 1.00 | 28.32 ± 12.76 | 1.018 ± 0.016 |
| LeWM | Mixed | 1 | 50.78 | 79.10 | 0.974 |
| LeWM | Original → Mixed Joint | 1 | 91.80 | 99.02 | 0.215 |
| LeWM | Original → Mixed Frozen Encoder | 1 | 97.46 | 98.83 | 0.128 |
| PLDM | Original | 3 | 45.57 ± 1.88 | 9.38 ± 4.20 | 1.058 ± 0.012 |
| PLDM | Mixed | 1 | 56.25 | 68.16 | 0.835 |
| PLDM | Original → Mixed Joint | 1 | 59.18 | 73.44 | 0.740 |
| PLDM | Original → Mixed Frozen Encoder | 1 | 98.63 | 99.80 | 0.101 |
| DINO-WM | Original | 1 | 38.48 | 9.38 | 1.127 |
| DINO-WM | Mixed | 1 | 97.66 | 99.61 | 0.394 |

## Arm mass

| Model | Training | n | Accuracy (%) ↑ | History (%) ↑ | NRE ↓ |
|---|---|---:|---:|---:|---:|
| LeWM | Original | 3 | 50.07 ± 0.11 | 51.63 ± 0.60 | 0.977 ± 0.002 |
| LeWM | Mixed | 1 | 85.74 | 89.84 | 0.045 |
| LeWM | Original → Mixed Joint | 1 | 87.30 | 93.36 | 0.038 |
| LeWM | Original → Mixed Frozen Encoder | 1 | 77.15 | 86.52 | 0.346 |
| PLDM | Original | 3 | 51.82 ± 1.30 | 66.99 ± 9.68 | 0.921 ± 0.016 |
| PLDM | Mixed | 1 | 91.02 | 93.55 | 0.031 |
| PLDM | Original → Mixed Joint | 1 | 84.77 | 90.82 | 0.073 |
| PLDM | Original → Mixed Frozen Encoder | 1 | 81.84 | 89.26 | 0.213 |
| DINO-WM | Original | 1 | 50.78 | 61.33 | 0.977 |
| DINO-WM | Mixed | 1 | 78.32 | 88.09 | 0.673 |

## Action delay

| Model | Training | n | Accuracy (%) ↑ | History (%) ↑ | NRE ↓ |
|---|---|---:|---:|---:|---:|
| LeWM | Mixed | 3 | 97.70 ± 0.59 | 99.18 ± 0.38 | 0.013 ± 0.002 |
| PLDM | Mixed | 3 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.006 ± 0.001 |
| DINO-WM | Mixed | 3 | 16.65 ± 0.03 | 19.45 ± 0.11 | 1.000 ± 0.000 |

## Contact friction

| Model | Training | n | Accuracy (%) ↑ | History (%) ↑ | NRE ↓ |
|---|---|---:|---:|---:|---:|
| LeWM | Original | 3 | 49.93 ± 0.23 | 50.13 ± 0.63 | 1.001 ± 0.004 |
| LeWM | Mixed | 1 | 50.00 | 49.41 | 1.000 |
| LeWM | Original → Mixed Joint | 1 | 50.39 | 54.30 | 0.999 |
| LeWM | Original → Mixed Frozen Encoder | 1 | 86.13 | 92.77 | 0.429 |
| PLDM | Original | 3 | 50.00 ± 0.34 | 49.09 ± 1.58 | 1.014 ± 0.010 |
| PLDM | Mixed | 1 | 50.00 | 48.63 | 1.001 |
| PLDM | Original → Mixed Joint | 1 | 50.00 | 46.48 | 1.020 |
| PLDM | Original → Mixed Frozen Encoder | 1 | 73.63 | 83.59 | 0.595 |
| DINO-WM | Original | 1 | 50.39 | 47.66 | 1.015 |
| DINO-WM | Mixed | 1 | 84.18 | 92.19 | 0.648 |

## Motion damping

| Model | Training | n | Accuracy (%) ↑ | History (%) ↑ | NRE ↓ |
|---|---|---:|---:|---:|---:|
| LeWM | Original | 3 | 50.00 ± 0.00 | 48.63 ± 1.60 | 1.007 ± 0.009 |
| LeWM | Mixed | 1 | 50.00 | 48.24 | 1.004 |
| LeWM | Original → Mixed Joint | 1 | 50.00 | 49.61 | 1.001 |
| LeWM | Original → Mixed Frozen Encoder | 1 | 53.91 | 61.91 | 0.958 |
| PLDM | Original | 3 | 50.26 ± 0.30 | 50.52 ± 0.63 | 0.947 ± 0.007 |
| PLDM | Mixed | 1 | 58.20 | 55.86 | 2.435 |
| PLDM | Original → Mixed Joint | 1 | 79.88 | 81.84 | 0.237 |
| PLDM | Original → Mixed Frozen Encoder | 1 | 54.49 | 64.26 | 0.867 |
| DINO-WM | Original | 1 | 49.80 | 49.61 | 1.014 |
| DINO-WM | Mixed | 1 | 56.45 | 65.23 | 1.042 |
| DINO-WM | Original → Mixed | 1 | 55.27 | 61.13 | 1.111 |

## Gripper carry rule

| Model | Training | n | Accuracy (%) ↑ | History (%) ↑ | NRE ↓ |
|---|---|---:|---:|---:|---:|
| LeWM | Original | 3 | 52.21 ± 1.11 | 73.76 ± 1.97 | 0.909 ± 0.029 |
| LeWM | Mixed | 1 | 50.00 | 49.80 | 1.000 |
| LeWM | Original → Mixed Joint | 1 | 50.00 | 52.34 | 0.997 |
| LeWM | Original → Mixed Frozen Encoder | 1 | 76.37 | 84.57 | 0.271 |
| PLDM | Original | 3 | 50.78 ± 0.20 | 57.42 ± 1.35 | 0.937 ± 0.007 |
| PLDM | Mixed | 1 | 49.80 | 50.78 | 0.997 |
| PLDM | Original → Mixed Joint | 1 | 50.20 | 54.30 | 0.982 |
| PLDM | Original → Mixed Frozen Encoder | 1 | 52.93 | 57.81 | 0.606 |
| DINO-WM | Original | 1 | 50.00 | 50.59 | 1.000 |
| DINO-WM | Mixed | 1 | 60.16 | 77.34 | 0.892 |
| DINO-WM | Original → Mixed | 1 | 62.89 | 80.27 | 0.871 |

## Door traversability

| Model | Training | n | Accuracy (%) ↑ | History (%) ↑ | NRE ↓ |
|---|---|---:|---:|---:|---:|
| LeWM | Original | 3 | 50.33 ± 0.00 | 50.94 ± 2.84 | 0.996 ± 0.003 |
| LeWM | Mixed | 3 | 99.83 ± 0.29 | 100.00 ± 0.00 | 0.041 ± 0.018 |
| LeWM | Original → Mixed Joint | 1 | 99.83 | 100.00 | 0.038 |
| LeWM | Original → Mixed Frozen Encoder | 1 | 100.00 | 100.00 | 0.036 |
| PLDM | Original | 3 | 50.06 ± 0.10 | 51.50 ± 0.67 | 0.997 ± 0.002 |
| PLDM | Mixed | 3 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.022 ± 0.005 |
| PLDM | Original → Mixed Joint | 1 | 100.00 | 100.00 | 0.023 |
| PLDM | Original → Mixed Frozen Encoder | 1 | 100.00 | 100.00 | 0.048 |
| DINO-WM | Original | 1 | 50.00 | 64.33 | 0.985 |
| DINO-WM | Mixed | 3 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.217 ± 0.003 |

## Portal exit

| Model | Training | n | Accuracy (%) ↑ | History (%) ↑ | NRE ↓ |
|---|---|---:|---:|---:|---:|
| LeWM | Original | 3 | 50.07 ± 0.11 | 51.95 ± 0.70 | 0.990 ± 0.003 |
| LeWM | Mixed | 1 | 50.78 | 76.37 | 0.922 |
| LeWM | Original → Mixed Joint | 1 | 50.39 | 66.80 | 0.934 |
| LeWM | Original → Mixed Frozen Encoder | 1 | 77.93 | 94.34 | 0.264 |
| PLDM | Original | 3 | 50.07 ± 0.11 | 51.50 ± 0.30 | 0.996 ± 0.001 |
| PLDM | Mixed | 1 | 72.27 | 79.10 | 0.128 |
| PLDM | Original → Mixed Joint | 1 | 52.54 | 61.52 | 0.708 |
| PLDM | Original → Mixed Frozen Encoder | 1 | 91.21 | 98.83 | 0.191 |
| DINO-WM | Original | 1 | 50.78 | 56.45 | 0.996 |
| DINO-WM | Mixed | 1 | 100.00 | 100.00 | 0.142 |

Source: [summary.json](summary.json). Machine readable values: [metrics.json](metrics.json) and [metrics.csv](metrics.csv).
