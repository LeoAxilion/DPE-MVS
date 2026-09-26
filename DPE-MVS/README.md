# DPE-MVS

## About

The paper has been released and can be found at [Dual-Level Precision Edges Guided Multi-View Stereo with Accurate Planarization](https://arxiv.org/abs/2412.20328).

If you find this project useful for your research, please cite:
>
    @article{chen2024dual,
        title={Dual-Level Precision Edges Guided Multi-View Stereo with Accurate Planarization},
        author={Chen, Kehua and Yuan, Zhenlong and Mao, Tianlu and Wang, Zhaoqi},
        journal={arXiv preprint arXiv:2412.20328},
        year={2024}
    }

## Dependencies
The code has been tested on Ubuntu 20.04 with Nvidia RTX 3090.

- [Cuda](https://developer.nvidia.cn/zh-cn/cuda-toolkit) >= 10.2
- [OpenCV](https://opencv.org/) >= 3.3.0
- [Boost](https://www.boost.org/) >= 1.62.0
- [cmake](https://cmake.org/) >= 2.8

**Besides make sure that your [GPU Compute Capability](https://en.wikipedia.org/wiki/CUDA) matches the CMakeList.txt!!!** Otherwise you won't get the depth results! For example, according to [GPU Compute Capability](https://en.wikipedia.org/wiki/CUDA), RTX3080's Compute Capability is 8.6. So you should set the cuda compilation parameter 'arch=compute_86,code=sm_86' or add a '-gencode arch=compute_86,code=sm_86'.

## Usage
- Compile
>
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j4

  Single-configuration builds default to `Release` if no build type is set.
  An explicitly selected `Debug` or other build type is respected. To print
  per-image CPU and synchronized PatchMatch stage wall times, run with
  `DPE_PROFILE=1 ./build/DPE ...`; profiling is disabled by default.

- Test
>
    Use script colmap2mvsnet_acm.py to convert COLMAP SfM result to MVS input
    Run ./DPE $data_folder to get reconstruction results.
    The result will be saved in the folder $data_folder/DPE, and the point cloud is saved as "DPE.ply"

- Fuse existing depth maps only
>
    ./DPE $data_folder 0 --fuse

  This mode skips edge detection and depth estimation. It requires `depths.dmb`,
  `normals.dmb`, and `weak.bin` in every `$data_folder/DPE/########/` directory.
  Fusion-only runs preserve these intermediate files so that fusion can be retried.

  Experimental plane-range fusion can be selected with:
>
    ./DPE $data_folder 0 --fuse --plane-fusion --plane-sample-stride 4

  It groups 4-connected, geometrically compatible pixels into plane patches,
  validates each patch against a few representative locations in neighboring
  views, then emits a regular grid of samples within accepted patch ranges.
  Samples keep their PatchMatch depth; non-planar, small, and unsupported areas
  use the original per-pixel fusion path. `--plane-sample-stride` controls the
  grid spacing in image pixels (larger values emit fewer points). This mode is
  experimental: on the meadow baseline it preserved ETH3D scores closely but
  reduced total fusion time by only about 1% across two runs, so it did not
  provide a substantial speedup. See [PLANE_FUSION_MEADOW_REPORT.md](PLANE_FUSION_MEADOW_REPORT.md).

  A normal depth-estimation run also preserves these three files after fusion;
  they are persistent fusion inputs and are independent of
  `PatchMatchParams::geometric_anchor_cost` and intermediate-visualization output.

- Experimental adaptive refinement and point sampling
>
    ./DPE $data_folder 0 --max-image-size 3200 --adaptive-refinement \
      --adaptive-point-sampling --simple-region-stride 2

  `--adaptive-refinement` keeps the depth maps dense but stops updating pixels
  whose completed PatchMatch confidence state is `STRONG`, or whose `WEAK`
  state has a post-anchor PatchMatch cost at most `0.15`. The confidence state
  is read after `DepthToWeak`, and the cost is read after anchor propagation
  and local refinement, so the initial random-plane
  setup is never used as a freeze decision. A qualifying pixel freezes
  immediately, and frozen pixels remain frozen in later iterations and are
  propagated to finer scales by freezing one representative pixel per frozen
  parent. The other pixels created by upsampling remain active with fresh
  state to recover fine detail, even when their parent was frozen.
  State is retained in memory between passes and checkpointed after each pyramid
  level, including images stopped early. `--start-round` reloads these completed
  level checkpoints. Neighbour generation, neighbour updating, RANSAC plane
  fitting, and local refinement use the active-pixel list once at least 25% of
  the pixels are frozen; below that threshold they use the full image grid.
  The legacy `--adaptive-refinement-aggressiveness` option is retained for
  command-line compatibility but no longer changes the freeze decision.
  The frozen mask stops depth/normal refinement and skips building an anchor
  list or fitting a plane for the frozen pixel itself. Frozen strong pixels
  remain available as anchors for active neighbours, and nearest-strong
  lookup is retained where an active pixel may read it. The weak/strong
  confidence state is still refreshed for fusion.
  Frozen pixels use equal weights over their previously selected views for
  this confidence check, since they do not regenerate propagation view weights.
  Once pixels are frozen, anchor-neighbour preparation, fitted-plane generation,
  and `LocalRefine` launch from a compact list of active pixels instead of
  assigning one CUDA thread to every image pixel. Checkerboard propagation
  retains its red/black launches; nearest-strong lookup and confidence refresh
  still cover the image because active pixels and fusion may need those values.
  Building the active list has a cost, so the speedup depends on how many
  pixels freeze and how much time these kernels occupy.
  `--adaptive-refinement-early-stop 0.85` optionally stops the remaining
  outer passes for one reference image at the current scale once at least 85%
  of its pixels are frozen. Active pixels then receive fewer updates, so this
  option is a stronger speed/quality trade-off; the default is disabled.
  `--adaptive-point-sampling` applies the
  same local plane test during fusion and samples accepted planar regions on a
  regular grid; stride 2 keeps one quarter of those planar candidates while
  retaining all candidates in locally complex areas. Both options are off by
  default. This is a speed/quality trade-off and should be checked against the
  unmodified result for each scene. Fusion-only mode supports the point-sampling
  options without rerunning depth estimation.

  `--start-round N` resumes from an already completed preceding pyramid scale.
  Use it only when every per-image depth, normal, weak-state, and selected-view
  file in the output folder belongs to the immediately preceding scale.

If you need to filter out the sky during point cloud fusion, you can use a segmentation approach. Please refer to [MP-MVS](https://github.com/RongxuanTan/MP-MVS) and save the segmentation results in the $data_folder/blocks directory.

## Acknowledgements
This code largely benefits from the following repositories: [APD-MVS](https://github.com/whoiszzj/APD-MVS), [HPM-MVS](https://github.com/CLinvx/HPM-MVS). Thanks to their authors for opening the source of their excellent works!
