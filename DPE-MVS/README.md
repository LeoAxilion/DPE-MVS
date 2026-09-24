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
    mkdir build & cd build
    cmake ..
    make

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

  A normal depth-estimation run also preserves these three files after fusion;
  they are persistent fusion inputs and are independent of
  `PatchMatchParams::geometric_anchor_cost` and intermediate-visualization output.

- Experimental adaptive refinement and point sampling
>
    ./DPE $data_folder 0 --max-image-size 3200 --adaptive-refinement \
      --adaptive-point-sampling --simple-region-stride 2

  `--adaptive-refinement` keeps the depth maps dense but stops updating pixels
  after a local 3x3 depth/normal neighbourhood agrees with one plane. The
  decision starts after the coarsest scale has produced its first depth map;
  a pixel must pass the plane test in two consecutive refinement checks before
  it freezes. Frozen pixels remain frozen in later iterations and are
  propagated to finer scales by mapping every child to its parent; if dimensions
  are rounded, the child freezes whenever its source footprint overlaps a frozen
  parent. Its optional
  `--adaptive-refinement-aggressiveness 1|2|3` setting
  selects conservative (8/8 neighbours, 15 degrees, 1.25% depth error, strong
  center only), balanced (6/8, 25 degrees, 3%, strong center only), or
  aggressive (5/8, 35 degrees, 5%, still requiring a strong center) consensus.
  Invalid depth and the image border remain active. The aggressive setting can
  freeze incorrect but locally smooth geometry, so compare depth maps and the
  final mesh against a conservative run before production use.
  The frozen mask stops depth/normal refinement while the weak/strong
  confidence state is still refreshed for fusion.
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
