# Geometric anchor cost (experimental)

Default remains original NCC. Separate executable: build-geometric-anchor/DPE.
Enable: DPE INPUT GPU --geometric-anchor-cost
Use separate prepared INPUT directories for baseline and experiment: normal execution
writes depth maps and deletes intermediate files after successful fusion.

For weak-pixel deformable NCC, keep the center patch. Replace all anchor patches
with one fitted-plane prior per candidate/view:
C = 0.25*NCC_center + 0.75*(E_normal + E_depth)
E_normal = min(1, (1-|dot(unit(n),unit(n_fit))|)/(1-cos(30deg)))
E_depth = min(1, |z-z_fit|/(0.01*z_fit)).
Both depths are evaluated at the center pixel in the reference camera.
The combined prior lies in [0,2], matching NCC range. Thresholds and weights are
initial experimental choices, not calibrated uncertainty estimates.

A separate GPU validity byte records actual RANSAC success. Without a valid fit,
or with invalid fitted depth/normal, use original anchor NCC. Invalid candidate
depth gets maximal geometric cost. Normal sign is ignored because (n,d) and
(-n,-d) describe the same plane. Strong-pixel NCC and fusion are unchanged.
The prior is currently recomputed per view (cheap arithmetic); center visibility
and center photometric evidence remain view dependent.

This replaces evidence with a prior; reliable anchors do not guarantee a correct
common plane. Nonplanar surfaces, occlusions and biased RANSAC fits can degrade.
No speedup or reconstruction improvement has been measured yet. Fusion OOM is
not addressed. Validate on separate copies with identical resolutions, pair lists,
seeds/settings and compare depth error/completeness plus stage runtime.

## Maximum image size

Default longest edge is now 3200; use `--max-image-size 0` for original resolution.
Example: `build-max-size/DPE INPUT 0 --geometric-anchor-cost --max-image-size 3200`.
Resizing is in memory, preserving original images/cameras. Never upscales. Intrinsics
use actual rounded width/height ratios. Pyramid edges and input processing use the
same capped dimensions; fusion rescales original color images/intrinsics to depth
map sizes and masks use nearest-neighbor resizing. Edge caches include the cap in
filenames. Outputs remain INPUT/DPE (back up previous results for comparison).
Existing same-size input requirement remains. Changing the limit requires rerunning
depth estimation, not mixing previous depth files. Build-max-size is separate so an
already-running batch continues using its existing executable.
