# Plane-range fusion: meadow comparison

## Method

`--plane-fusion` builds 4-connected plane patches from the saved depth and normal maps. A pixel can join the seed patch when its normal differs by at most 8 degrees and its 3D point lies within `max(5 mm, 0.2% of seed depth)` of the seed plane. Patches smaller than 16 pixels remain on the original point-fusion path.

For an accepted patch, the implementation checks five representative locations against each neighboring image. A neighboring patch counts as support when at least two representatives land on a patch with a compatible plane. It then samples the accepted image range on a regular pixel grid. Output sample locations retain their original PatchMatch depths. Unsupported patches and unsegmented areas use the original per-pixel consistency test. The default stride is 4; `--plane-sample-stride N` changes the spacing and resulting point density.

## Evaluation setup

Both methods used the same already-computed non-adaptive meadow depth, normal, and weak-state files, so the comparison measures fusion only and does not include depth estimation. ETH3D tolerances were `0.01,0.02,0.05,0.1,0.2,0.5` m. Two timed fusion runs were made per method. Repeated runs produced identical PLY hashes within each method.

| Method | Fusion elapsed, run 1 / run 2 | Mean | Fused points | Peak host RSS |
| --- | ---: | ---: | ---: | ---: |
| Original pixel fusion | 81.87 / 82.04 s | 81.96 s | 3,824,976 | 2.30 GB |
| Plane-range fusion, stride 4 | 80.95 / 81.39 s | 81.17 s | 3,653,413 | 2.73 GB |

The measured mean reduction is 0.79 s, or 0.96%. This is too small to claim a substantial or stable speedup. Plane segmentation took about 9.8 s; the sampled patch path reduced repeated point-level view checks, but segmentation and range processing used most of that saving. Peak host memory rose by about 0.4 GB.

## ETH3D results

Values are percentages. Each plane-fusion cell includes the difference from pixel fusion in percentage points.

| Tolerance | Completeness | Accuracy | F1 |
| --- | ---: | ---: | ---: |
| 0.01 m | 48.278% (+1.578 pp) | 77.733% (-0.192 pp) | 59.563% (+1.162 pp) |
| 0.02 m | 76.695% (+0.836 pp) | 88.691% (+0.209 pp) | 82.258% (+0.572 pp) |
| 0.05 m | 91.277% (+0.103 pp) | 95.161% (+0.529 pp) | 93.178% (+0.308 pp) |
| 0.10 m | 93.581% (+0.042 pp) | 97.530% (+0.382 pp) | 95.514% (+0.205 pp) |
| 0.20 m | 95.108% (+0.062 pp) | 99.020% (+0.022 pp) | 97.025% (+0.043 pp) |
| 0.50 m | 97.620% (-0.002 pp) | 99.555% (-0.041 pp) | 98.578% (-0.021 pp) |

The stride-4 result kept the six F1 scores close to or slightly above baseline, while reducing the emitted point count by about 4.5%. At 1 cm, accuracy fell by 0.19 percentage points; at the other tolerances the accuracy changes were small and positive except at 50 cm.

Stride 2 was also evaluated once. It emitted 6,209,202 points and took 80.60 s. Its 1 cm and 2 cm F1 scores were 62.751% and 82.905%, respectively; stride 4 used fewer points and had slightly better accuracy at 1 cm. Since total time was nearly unchanged, stride 4 is the more balanced setting from these runs.

## Artifacts

The raw runs and ETH3D logs are in `/media/media01/lxiao/ETH3D-preprocessed-DPE/multi_view_training_dslr_undistorted/meadow-plane-fusion-20260926/`. Reproduce the selected case with `bash run-patch.sh 4 patch-masked-stride4` from that directory.
