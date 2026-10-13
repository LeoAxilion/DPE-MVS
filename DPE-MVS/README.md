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

  Defaults: the longest image edge is capped at 1800 pixels, geometry-aware
  adaptive freezing is enabled, and final fusion retains one representative
  point for each frozen planar parent while keeping all points in active detail
  regions. Use `--max-image-size N` to override the cap (`0` keeps the input
  resolution). Use `--no-adaptive-geometry-density` to disable geometry-
  consistent freezing and its density-aware fusion behavior.

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
  In particular, every reference image keeps its final per-image depth map at
  `$data_folder/DPE/########/depths.dmb`, even when intermediate visualization
  images are not requested. `colmap2mvsnet.py` writes
  `$data_folder/dpe_view_index.json` to map each zero-based DPE index back to
  its COLMAP image ID and source image name. These maps can be used later for
  source-image visibility checks without rerunning PatchMatch.

- Geometry-aware adaptive refinement and optional point sampling
>
    ./DPE $data_folder 0 --adaptive-point-sampling --simple-region-stride 2

  Geometry-consistent freezing is enabled by default. At each scale boundary,
  STRONG and WEAK pixels are candidates, but freeze only when their plane agrees
  with the required neighbors and their depth is in range;
  default aggressiveness 1 requires 8/8 neighbors, normal differences at most
  12 degrees, and relative depth error at most 1.25%. Frozen regions remain
  frozen at finer scales. Their child regions are not refined, and final fusion
  keeps one representative point per frozen planar parent while retaining all
  candidates in active, geometrically complex regions. This preserves dense
  detail near depth changes while reducing redundant planar samples.

  `--no-adaptive-geometry-density` disables all freezing and density-aware
  fusion, restoring the standard dense path. The former
  `--adaptive-geometry-density` enable flag has been replaced by this
  default-on opt-out. `--no-adaptive-refinement` remains as a legacy alias;
  `--adaptive-refinement` remains accepted as a no-op for old command lines.
  Confidence-only freezing has been removed. In fusion-only mode,
  geometry-aware fusion needs
  `adaptive_density_keep.dmb` files produced by a matching adaptive run; pass
  `--no-adaptive-geometry-density` to fuse older depth maps with the standard path.

  Compact active-pixel GPU worklists are enabled by default, so frozen pixels
  are omitted from per-pixel PatchMatch launches, including initialization and
  checkerboard kernels. Use `--no-compact-active-worklists` to disable
  compaction and restore full-grid thread assignment.

  The confidence state is read after `DepthToWeak`; the initial random-plane
  setup is never used as a freeze decision. Frozen pixels are terminal in the
  compact path and keep their last state. State is retained in memory between
  passes and checkpointed after each pyramid level, including images stopped
  early. `--start-round` reloads these completed level checkpoints. Neighbour
  generation, neighbour updating, RANSAC plane fitting, local refinement,
  checkerboard propagation, and confidence updates use active-pixel worklists
  whenever frozen pixels exist. Disabling compaction restores full-grid launches.
  `--adaptive-refinement-aggressiveness` controls the plane-consensus threshold:
  level 1 requires 8/8 neighbors, 12-degree normal tolerance, and 1.25% relative
  depth tolerance; levels 2 and 3 relax these thresholds.
  The frozen mask stops depth/normal refinement and skips building an anchor
  list or fitting a plane for the frozen pixel itself. Frozen strong pixels
  remain available as anchors for active neighbours, and nearest-strong
  lookup is retained where an active pixel may read it. Frozen pixels retain
  their last weak/strong state because they no longer receive confidence-update
  threads.
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
  retaining all candidates in locally complex areas. This additional sampling
  option remains off by default; the geometry-aware frozen-parent fusion above
  is on by default. This is a speed/quality trade-off and should be checked
  against the unmodified result for each scene. Fusion-only mode supports the
  point-sampling option without rerunning depth estimation.

  `--start-round N` resumes from an already completed preceding pyramid scale.
  Use it only when every per-image depth, normal, weak-state, and selected-view
  file in the output folder belongs to the immediately preceding scale.

  `--min-pyramid-levels N` keeps at least `N` pyramid levels even when the
  image-size rule would normally produce fewer. `--pyramid-levels N` instead
  selects exactly `N` levels, including the final full-resolution level capped
  by `--max-image-size`; it is mutually exclusive with
  `--min-pyramid-levels`. For example, with `--max-image-size 800`,
  `--pyramid-levels 3` processes 200, 400, and 800 pixels along the longest
  image dimension.

If you need to filter out the sky during point cloud fusion, you can use a segmentation approach. Please refer to [MP-MVS](https://github.com/RongxuanTan/MP-MVS) and save the segmentation results in the $data_folder/blocks directory.

## Acknowledgements
This code largely benefits from the following repositories: [APD-MVS](https://github.com/whoiszzj/APD-MVS), [HPM-MVS](https://github.com/CLinvx/HPM-MVS). Thanks to their authors for opening the source of their excellent works!
