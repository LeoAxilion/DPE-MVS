#ifndef _MAIN_H_
#define _MAIN_H_

// Includes Opencv
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>
// Includes STD libs
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <cstdarg>
#include <random>
#include <unordered_map>
// Includes Boost filesystem
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>


// Define some const var
#define OUT_NAME "DPE"
#define MAX_IMAGES 32
#define NEIGHBOUR_NUM 9
#define SURF_COEFF_NUM 10
#define MAX_SEARCH_RADIUS 4096
#define DEBUG_POINT_X 753
#define DEBUG_POINT_Y 259
//#define DEBUG_COST_LINE
//#define DEBUG_NEIGHBOUR

using namespace boost::filesystem;

struct Camera {
	float K[9];
	float R[9];
	float t[3];
	float c[3];
	int height;
	int width;
	float depth_min;
	float depth_max;
};

struct PointList {
	float3 coord;
	float3 color;
	// World-space reference normal retained for fused.ply.
	float3 normal = { 0.0f, 0.0f, 0.0f };
};

enum RunState {
	FIRST_INIT,
	REFINE_INIT,
	REFINE_ITER
};

enum PixelState {
	WEAK,
	STRONG,
	UNKNOWN
};

struct PatchMatchParams {
	int max_iterations = 3;
	int initial_image_scale = 4; // N scales both image dimensions to approximately 1/N.
	bool geometric_anchor_cost = false; // Opt-in experimental replacement of anchor NCC.
	bool adaptive_refinement = false; // Skip fine-scale PatchMatch updates on stable planar strong pixels.
	int adaptive_refinement_aggressiveness = 1; // 1=conservative, 2=balanced, 3=aggressive planar consensus.
	// Freeze complete flat parent regions at the next pyramid level and keep
	// one representative point per frozen region during final fusion.
	bool adaptive_geometry_density = false;
	// Compact all checkerboard/active kernels to the unfrozen region. This is
	// the regional pyramid mode: frozen planar children remain logical leaves
	// and do not receive PatchMatch work at the finer scale.
	bool regional_adaptive_pyramid = false;
	int num_images = 5;
	float sigma_spatial = 5.0f;
	float sigma_color = 3.0f;
	int top_k = 4;	// ETH3D: 4 TAT: 8
	float depth_min = 0.0f;
	float depth_max = 1.0f;
	bool geom_consistency = false;
	int strong_radius = 5;
	int strong_increment = 2;
	int weak_radius = 5;
	int weak_increment = 5;
	bool use_APD = true;
	//=====================================================
	bool use_edge = true;		// 强像素的感受野扩展（采样）
	bool use_limit = true;  	// 弱像素三角构成的边缘限制
	bool use_label = true;		// 弱像素的感受野扩展（采样）
	bool use_radius = true;		// 自适应调整弱像素匹配代价计算策略
	bool high_res_img = true;  // ETH：true TAT：false（在ETH3D这种大分辨率上效果好，说明尽量不要用小分辨率）
	int max_scale_size = 1;
	int scale_size = 1;
	//=====================================================
	int weak_peak_radius = 2;
	int rotate_time = 4;
	float ransac_threshold = 0.005;
	float geom_factor = 0.2f;
	RunState state;
};

struct AdaptiveRefinementState {
	cv::Mat mask;
	cv::Mat stability;
	// One representative sample per frozen parent region at the output scale.
	cv::Mat density_keep;
};

class ScaledImageCache;

struct Problem {
	int index;
	int ref_image_id;
	std::vector<int> src_image_ids;
	path dense_folder;
	path result_folder;
	int scale_size = 1;
	PatchMatchParams params;
	bool show_medium_result;
	int iteration;
	// Geometry-aware density mode updates its mask only at the end of a scale;
	// the final scale keeps the inherited mask for sparse fusion sampling.
	bool update_adaptive_mask = true;
	std::shared_ptr<ScaledImageCache> scaled_image_cache;
	// Shared by successive DPE instances for this image; checkpointed per scale.
	std::shared_ptr<AdaptiveRefinementState> adaptive_state =
		std::make_shared<AdaptiveRefinementState>();
};

// Per-process cache for the already prepared image files of the active
// pyramid scale. Retain() drops views not used by the next ProcessProblem.
class ScaledImageCache {
public:
	explicit ScaledImageCache(size_t capacity);
	void Retain(const std::vector<int> &image_ids);
	void SetOriginalSize(int image_id, const cv::Size &size);
	cv::Size OriginalSize(int image_id) const;
	cv::Mat Get(int image_id, const path &scaled_image_path);
	void Clear();
	void ResetStats();
	size_t CacheHits() const { return cache_hits; }
	size_t CacheMisses() const { return cache_misses; }
	double DiskReadSeconds() const { return disk_read_seconds; }
	size_t Capacity() const { return capacity; }

private:
	size_t capacity;
	size_t cache_hits = 0;
	size_t cache_misses = 0;
	double disk_read_seconds = 0.0;
	std::unordered_map<int, cv::Mat> images;
	std::unordered_map<int, cv::Size> original_sizes;
};

path ScaledImagePath(const path &dense_folder, int image_id,
	int initial_image_scale, int pyramid_scale);

// Shared sizing rule for images, intrinsics, and edge pyramids. N=1 keeps
// the input dimensions; N=4 scales width and height to approximately 1/4.
inline cv::Size ScaledImageSize(int width, int height, int scale) {
    if (scale <= 1) return cv::Size(width, height);
    return cv::Size(
        std::max(1, static_cast<int>(std::round(width / static_cast<double>(scale)))),
        std::max(1, static_cast<int>(std::round(height / static_cast<double>(scale)))));
}

// Preserve existing pyramid dimensions while resizing directly from the
// original image to one pyramid level in a single interpolation pass.
inline cv::Size PyramidLevelImageSize(int width, int height, int initial_scale,
	int pyramid_scale) {
	const cv::Size base = ScaledImageSize(width, height, initial_scale);
	if (pyramid_scale <= 1) return base;
	return cv::Size(
		std::max(1, static_cast<int>(std::round(base.width / static_cast<double>(pyramid_scale)))),
		std::max(1, static_cast<int>(std::round(base.height / static_cast<double>(pyramid_scale)))));
}

#endif // !_MAIN_H_
