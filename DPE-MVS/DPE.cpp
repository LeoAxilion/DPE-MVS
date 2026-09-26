#include "DPE.h"
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

#define _JACOBI_ROTATE(a, i, j, k, l) \
	g = (a)[j][i]; \
	h = (a)[l][k]; \
	(a)[j][i] = g - s * (h + g * tau); \
	(a)[l][k] = h + s *(g - h * tau);

namespace {
bool IsFiniteFloat(float value) {
	std::uint32_t bits = 0;
	static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
	std::memcpy(&bits, &value, sizeof(bits));
	return (bits & 0x7f800000u) != 0x7f800000u;
}

bool IsValidNormal(const cv::Vec3f &normal) {
	if (!IsFiniteFloat(normal[0]) || !IsFiniteFloat(normal[1]) || !IsFiniteFloat(normal[2])) return false;
	const float length_squared = normal.dot(normal);
	return IsFiniteFloat(length_squared) && length_squared > 1e-12f;
}

// Preserve one representative of each frozen pixel at finer pyramid levels.
// Newly created pixels stay active so they can resolve finer-scale details.
// On downsampling, retain the conservative footprint behavior.
int RepresentativeCoordinate(int source_coordinate, int source_length, int target_length) {
	return static_cast<int>(((2LL * source_coordinate + 1) * target_length) /
		(2LL * source_length));
}

cv::Mat PropagateFrozenMask(const cv::Mat &source, const cv::Size &target_size) {
	CV_Assert(source.type() == CV_8UC1 && target_size.width > 0 && target_size.height > 0);
	cv::Mat target(target_size, CV_8UC1, cv::Scalar(1));
	if (target.cols >= source.cols && target.rows >= source.rows) {
		for (int y = 0; y < source.rows; ++y) {
			const uchar *source_row = source.ptr<uchar>(y);
			const int target_y = RepresentativeCoordinate(y, source.rows, target.rows);
			uchar *target_row = target.ptr<uchar>(target_y);
			for (int x = 0; x < source.cols; ++x) {
				if (source_row[x] == 0) {
					const int target_x = RepresentativeCoordinate(x, source.cols, target.cols);
					target_row[target_x] = 0;
				}
			}
		}
		return target;
	}
	for (int y = 0; y < target.rows; ++y) {
		const int source_y_begin = y * source.rows / target.rows;
		const int source_y_end = std::min(source.rows,
			((y + 1) * source.rows + target.rows - 1) / target.rows);
		for (int x = 0; x < target.cols; ++x) {
			const int source_x_begin = x * source.cols / target.cols;
			const int source_x_end = std::min(source.cols,
				((x + 1) * source.cols + target.cols - 1) / target.cols);
			bool frozen = false;
			for (int sy = source_y_begin; sy < std::max(source_y_begin + 1, source_y_end) && !frozen; ++sy) {
				const uchar *source_row = source.ptr<uchar>(std::min(sy, source.rows - 1));
				for (int sx = source_x_begin; sx < std::max(source_x_begin + 1, source_x_end); ++sx) {
					if (source_row[std::min(sx, source.cols - 1)] == 0) {
						frozen = true;
						break;
					}
				}
			}
			if (frozen) target.at<uchar>(y, x) = 0;
		}
	}
	return target;
}

// Stability belongs to the tested pixel, not to every pixel interpolated from it.
cv::Mat PropagateStabilityCount(const cv::Mat &source, const cv::Size &target_size) {
	CV_Assert(source.type() == CV_8UC1 && target_size.width > 0 && target_size.height > 0);
	if (target_size.width < source.cols || target_size.height < source.rows) {
		cv::Mat target;
		cv::resize(source, target, target_size, 0, 0, cv::INTER_NEAREST);
		return target;
	}
	cv::Mat target(target_size, CV_8UC1, cv::Scalar(0));
	for (int y = 0; y < source.rows; ++y) {
		const uchar *source_row = source.ptr<uchar>(y);
		uchar *target_row = target.ptr<uchar>(RepresentativeCoordinate(y, source.rows, target.rows));
		for (int x = 0; x < source.cols; ++x)
			target_row[RepresentativeCoordinate(x, source.cols, target.cols)] = source_row[x];
	}
	return target;
}

bool IsStablePlanarNeighbourhood(const cv::Mat &depth, const cv::Mat &normal,
		const cv::Mat &edge, const Camera &camera, int r, int c,
		int min_agreements = 8, float max_normal_angle_degrees = 15.0f,
		float max_relative_depth_error = 0.0125f) {
	if (r < 1 || c < 1 || r + 1 >= depth.rows || c + 1 >= depth.cols ||
		depth.type() != CV_32FC1 || normal.type() != CV_32FC3) return false;
	if (!edge.empty() && edge.type() == CV_8UC1) {
		for (int dy = -1; dy <= 1; ++dy)
			for (int dx = -1; dx <= 1; ++dx)
				if (edge.at<uchar>(r + dy, c + dx) != 0) return false;
	}
	const float z0 = depth.at<float>(r, c);
	const cv::Vec3f n0w = normal.at<cv::Vec3f>(r, c);
	const float n0len = std::sqrt(n0w.dot(n0w));
	if (!IsFiniteFloat(z0) || z0 <= 0.0f || !IsFiniteFloat(n0len) || n0len < 1e-6f) return false;
	const cv::Vec3f n0 = n0w * (1.0f / n0len);
	const cv::Vec3f n0c(
		camera.R[0] * n0[0] + camera.R[1] * n0[1] + camera.R[2] * n0[2],
		camera.R[3] * n0[0] + camera.R[4] * n0[1] + camera.R[5] * n0[2],
		camera.R[6] * n0[0] + camera.R[7] * n0[1] + camera.R[8] * n0[2]);
	const float plane_constant = z0 * (n0c[0] * (c - camera.K[2]) / camera.K[0] +
		n0c[1] * (r - camera.K[5]) / camera.K[4] + n0c[2]);
	if (!IsFiniteFloat(plane_constant) || std::fabs(plane_constant) < 1e-8f) return false;
	const float kNormalCosine = std::cos(max_normal_angle_degrees * static_cast<float>(M_PI) / 180.0f);
	int agreements = 0;
	for (int dy = -1; dy <= 1; ++dy) {
		for (int dx = -1; dx <= 1; ++dx) {
			if (dx == 0 && dy == 0) continue;
			const int nr = r + dy, nc = c + dx;
			const float zn = depth.at<float>(nr, nc);
			const cv::Vec3f nnw = normal.at<cv::Vec3f>(nr, nc);
			const float nnlen = std::sqrt(nnw.dot(nnw));
			if (!IsFiniteFloat(zn) || zn <= 0.0f || !IsFiniteFloat(nnlen) || nnlen < 1e-6f) continue;
			const cv::Vec3f nn = nnw * (1.0f / nnlen);
			const float normal_cosine = n0.dot(nn);
			if (!IsFiniteFloat(normal_cosine) || std::fabs(normal_cosine) < kNormalCosine) continue;
			const float ray_dot = n0c[0] * (nc - camera.K[2]) / camera.K[0] +
				n0c[1] * (nr - camera.K[5]) / camera.K[4] + n0c[2];
			if (!IsFiniteFloat(ray_dot) || std::fabs(ray_dot) < 1e-6f) continue;
			const float predicted = plane_constant / ray_dot;
			if (!IsFiniteFloat(predicted) || predicted <= 0.0f) continue;
			const float relative_error = std::fabs(predicted - zn) / zn;
			if (!IsFiniteFloat(relative_error) || relative_error > max_relative_depth_error) continue;
			++agreements;
		}
	}
	return agreements >= min_agreements;
}
}

cv::Mat Roberts(const cv::Mat& src_image) {
	cv::Mat dst_image = src_image.clone();
	for (int i = 0; i < dst_image.rows; i++) {
		for (int j = 0; j < dst_image.cols; j++) {
			int t1 = 0, t2 = 0;
			if (i > 0 && i < dst_image.rows - 1 && j > 0 && j < dst_image.cols - 1) {
				t1 = (src_image.at<uchar>(i, j) - src_image.at<uchar>(i + 1, j + 1));
				t2 = (src_image.at<uchar>(i + 1, j) - src_image.at<uchar>(i, j + 1));
			}
			else {
				t1 = t2 = 50;
			}
			dst_image.at<uchar>(i, j) = (uchar)sqrt(t1 * t1 + t2 * t2);
		}
	}
	return dst_image;
}

// 求连通区域
void Connect(const cv::Mat& dst_image, cv::Mat &label_mask, std::vector<int> &label_cnt) {
	std::vector<std::vector<int>> left_neigh(dst_image.rows);
	std::vector<std::vector<int>> up_neigh(dst_image.rows);

	for (int y = 0; y < dst_image.rows; y++) {
		left_neigh[y].resize(dst_image.cols);
		up_neigh[y].resize(dst_image.cols);
		for (int x = 0; x < dst_image.cols; x++) {
			// 左连通
			if (x == 0) {
				left_neigh[y][x] = 0;
			} else {
				if (dst_image.at<uchar>(y, x) == 0 && dst_image.at<uchar>(y, x - 1) == 0) {
					left_neigh[y][x] = 1;
				} else {
					left_neigh[y][x] = 0;
				}
			}
			// 上连通
			if (y == 0) {
				up_neigh[y][x] = 0;
			} else {
				if (dst_image.at<uchar>(y, x) == 0 && dst_image.at<uchar>(y - 1, x) == 0) {
					up_neigh[y][x] = 1;
				} else {
					up_neigh[y][x] = 0;
				}
			}
		}
	}

	// 维护一个并查集
	int cnt = 1;
	std::vector<int> connection;
	connection.push_back(0);
	for (int y = 0; y < dst_image.rows; y++) {
		for (int x = 0; x < dst_image.cols; x++) {
			if (dst_image.at<uchar>(y, x) == 255) {
				label_mask.at<int>(y, x) = 0;
			} else {
				bool left = false, up = false;
				if (left_neigh[y][x] == 1) {
					label_mask.at<int>(y, x) = label_mask.at<int>(y, x - 1);
					left = true;
				}
				if (up_neigh[y][x] == 1) {
					label_mask.at<int>(y, x) = label_mask.at<int>(y - 1, x);
					up = true;
				}
				if (left == false && up == false) {
					label_mask.at<int>(y, x) = cnt;
					connection.push_back(cnt);
					cnt++;
				} else if (left == true && up == true) {
					int left_label = label_mask.at<int>(y, x - 1);
					int up_label = label_mask.at<int>(y - 1, x);
					if (left_label > up_label) {
						connection[left_label] = up_label;
						label_mask.at<int>(y, x) = label_mask.at<int>(y - 1, x);
					} else if (left_label < up_label) {
						connection[up_label] = left_label;
						label_mask.at<int>(y, x) = label_mask.at<int>(y, x - 1);
					}
				}
			}
		}
	}

	for (size_t i = 1; i < connection.size(); i++) {
		int cur_label = connection[i];
		int pre_label = connection[cur_label];
		while (pre_label != cur_label) {
			cur_label = pre_label;
			pre_label = connection[pre_label];
		}
		connection[i] = cur_label;
	}

	int label_num = 1;
	std::vector<int> mapping;
	mapping.push_back(0);
	for (size_t i = 1; i < connection.size(); i++) {
		mapping.push_back(0);
		if (connection[i] == (int)i) {
			mapping[i] = label_num;
			label_num++;  //标签总数
		}
	}

	// 重编号
	for (size_t i = 1; i < connection.size(); i++) {
		connection[i] = mapping[connection[i]];
	}

	for (int i = 0; i < label_num; i++) {
		label_cnt.push_back(0);
	}

	// 连通区域计数
	for (int y = 0; y < dst_image.rows; y++) {
		for (int x = 0; x < dst_image.cols; x++) {
			int label = label_mask.at<int>(y, x);
			label_mask.at<int>(y, x) = connection[label];
			label_cnt[connection[label]]++;
		}
	}
}

cv::Mat EdgeSegment(const int scale, const cv::Mat& src_image, int mode, bool use_canny, bool high_res_img) {
	/*
		mode - Edge:0; Label:1; Segmentation Image:2
	*/
	const int robthr = high_res_img ? 4 : 6;
	const int weak_tex_num = (int)(1.0 * src_image.rows * src_image.cols / (1024 << scale << scale));
	cv::Mat src_down;
	if (high_res_img) {
		cv::resize(src_image, src_down, cv::Size(src_image.cols / 2, src_image.rows / 2), 0, 0, cv::INTER_LINEAR);
	} else {
		src_down = src_image.clone();
	}

    cv::Mat dst_image;
	if (!use_canny) {
		cv::resize(src_down, src_down, cv::Size(src_down.cols / 2, src_down.rows / 2), 0, 0, cv::INTER_LINEAR);

		const int houthr = (int)MIN(src_down.cols, src_down.rows) / 30.0;
		const int min_line_length = (int)MIN(src_down.cols, src_down.rows) / 30.0;
		const int max_line_gap = (int)MIN(src_down.cols, src_down.rows) / 30.0;

		dst_image = Roberts(src_down);
		cv::threshold(dst_image, dst_image, robthr, 255, cv::THRESH_BINARY);

		cv::Mat lab_mask0(dst_image.rows, dst_image.cols, CV_32S);
		std::vector<int> label_cnt0;
		Connect(dst_image, lab_mask0, label_cnt0);

		for (size_t k = 1; k < label_cnt0.size(); k++) {
			if (label_cnt0[k] < weak_tex_num) continue;
			int weak_index = k;
			cv::Mat img_weak(dst_image.rows, dst_image.cols, CV_8UC1, cv::Scalar(0)); // 直接创建单通道二值图像

			for (int y = 0; y < img_weak.rows; y++) {
				for (int x = 0; x < img_weak.cols; x++) {
					int label = lab_mask0.at<int>(y, x);
					if (label == weak_index) continue;

					bool border = false;
					if (x > 0 && lab_mask0.at<int>(y, x - 1) == weak_index) border = true;
					if (x < img_weak.cols - 1 && lab_mask0.at<int>(y, x + 1) == weak_index) border = true;
					if (y > 0 && lab_mask0.at<int>(y - 1, x) == weak_index) border = true;
					if (y < img_weak.rows - 1 && lab_mask0.at<int>(y + 1, x) == weak_index) border = true;

					if (border)
						img_weak.at<uchar>(y, x) = 255; // 设置边界像素为白色
				}
			}

			std::vector<cv::Vec4i> lines;
			cv::HoughLinesP(img_weak, lines, 1, CV_PI / 180, houthr, min_line_length, max_line_gap); // 直接使用二值图像
			for (size_t i = 0; i < lines.size(); i++) {
				cv::line(dst_image, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0, 0), 1); // 修改为绘制蓝色线条
			}
		}
	} else {
		// 求像素中值
		int rows = src_image.rows;
		int cols = src_image.cols;
		int median_val = -1;
		float histogram[256] = { 0 };
		//先计算图像的直方图
		for (int i = 0; i < rows; ++i)
		{
			///获取i行首像素的指针
			const uchar *p = src_image.ptr<uchar>(i);
			///遍历i行像素
			for (int j = 0; j < cols; ++j) {
				histogram[int(*p++)]++;
			}
		}
		int HalfNum = rows * cols / 2;
		int tempSum = 0;
		for (int i = 0; i < 255; i++) {
			tempSum = tempSum + histogram[i];
			if (tempSum > HalfNum) {
				median_val = i;
				break;
			}
		}
		
		const float sigma = 0.67;
		int threshold1 = (1 - sigma) * median_val;
		int threshold2 = median_val;

		cv::Canny(src_image, dst_image, threshold1, threshold2, 3, true);
	}

	if (mode == 0) {
		cv::resize(dst_image, dst_image, cv::Size(src_image.cols, src_image.rows), 0, 0, cv::INTER_LINEAR);
	} else {
		const float factor = 1.0f / (float)(1 << scale);
		const int new_cols = std::round(src_image.cols * factor);
		const int new_rows = std::round(src_image.rows * factor);
		cv::resize(dst_image, dst_image, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
	}

	cv::threshold(dst_image, dst_image, robthr, 255, cv::THRESH_BINARY);

    cv::Mat label_mask(dst_image.rows, dst_image.cols, CV_32S);
    std::vector<int> label_cnt;
    std::vector<int> weakLabel;

    for (int y = 0; y < dst_image.rows; y++) {
		if (dst_image.data[y * dst_image.cols + 1] == 0)
			dst_image.data[y * dst_image.cols] = 0;
		if (dst_image.data[y * dst_image.cols + dst_image.cols - 2] == 0)
			dst_image.data[y * dst_image.cols + dst_image.cols - 1] = 0;
	}
	for (int x = 0; x < dst_image.cols; x++) {
		if (dst_image.data[1 * dst_image.cols + x] == 0)
			dst_image.data[0 * dst_image.cols + x] = 0;
		if (dst_image.data[(dst_image.rows - 2) * dst_image.cols + x] == 0)
			dst_image.data[(dst_image.rows - 1) * dst_image.cols + x] = 0;
	}

	if (mode == 0) {
		return dst_image;
	}

    Connect(dst_image, label_mask, label_cnt);

    int label_num = label_cnt.size();
    std::vector<cv::Vec3b> colors(label_num);
    colors[0] = cv::Vec3b(0, 0, 0);
    int label_cnt_max = 0;
    for (int i = 1; i < label_num; i++) {
        label_cnt_max = MAX(label_cnt_max, label_cnt[i]);
		colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
    }

    cv::Mat img_connect(dst_image.rows, dst_image.cols, CV_8UC3);
    for (int y = 0; y < img_connect.rows; y++) {
        for (int x = 0; x < img_connect.cols; x++) {
            img_connect.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            int label = label_mask.at<int>(y, x);
			// if (label_cnt[label] > weak_tex_num) {
            // 	img_connect.at<cv::Vec3b>(y, x) = colors[label];
			// } else {
			// 	if (label != 0) {
			// 		label_mask.at<int>(y, x) = -1;
			// 	}
			// }
			img_connect.at<cv::Vec3b>(y, x) = colors[label];
			if (label_cnt[label] <= weak_tex_num && label != 0) {
				label_mask.at<int>(y, x) = -1;
			}
        }
    }
	
	if (mode == 1) {
		return label_mask;
	}

	return img_connect;
}

bool ReadBinMat(const path &mat_path, cv::Mat &mat)
{
	ifstream in(mat_path, std::ios_base::binary);
	if (in.bad()) {
		std::cerr << "Error opening file: " << mat_path << std::endl;
		return false;
	}

	int version, rows, cols, type;
	in.read((char *)(&version), sizeof(int));
	in.read((char *)(&rows), sizeof(int));
	in.read((char *)(&cols), sizeof(int));
	in.read((char *)(&type), sizeof(int));

	if (version != 1) {
		in.close();
		std::cerr << "Version error: " << mat_path << std::endl;
		return false;
	}

	mat = cv::Mat(rows, cols, type);
	in.read((char *)mat.data, sizeof(char) * mat.step * mat.rows);
	in.close();
	return true;

}

bool WriteBinMat(const path &mat_path, const cv::Mat &mat) {

	ofstream out(mat_path, std::ios_base::binary);
	if (out.bad()) {
		std::cout << "Error opening file: " << mat_path << std::endl;
		return false;
	}
	int version = 1;
	int rows = mat.rows;
	int cols = mat.cols;
	int type = mat.type();

	out.write((char *)&version, sizeof(int));
	out.write((char *)&rows, sizeof(int));
	out.write((char *)&cols, sizeof(int));
	out.write((char *)&type, sizeof(int));
	out.write((char *)mat.data, sizeof(char) * mat.step * mat.rows);
	out.close();
	return true;
}

bool ReadCamera(const path &cam_path, Camera &cam)
{
	ifstream in(cam_path);
	if (in.bad()) {
		return false;
	}

	std::string line;
	in >> line;

	for (int i = 0; i < 3; ++i) {
		in >> cam.R[3 * i + 0] >> cam.R[3 * i + 1] >> cam.R[3 * i + 2] >> cam.t[i];
	}

	float tmp[4];
	in >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3];
	in >> line;

	for (int i = 0; i < 3; ++i) {
		in >> cam.K[3 * i + 0] >> cam.K[3 * i + 1] >> cam.K[3 * i + 2];
	}
	// compute camera center in world coord
	const auto &R = cam.R;
	const auto &t = cam.t;
	for (int j = 0; j < 3; ++j) {
		cam.c[j] = -float(double(R[0 + j])*double(t[0]) + double(R[3 + j])*double(t[1]) + double(R[6 + j])*double(t[2]));
	}
	// ====================================================================
	// TAT & ETH version read
	float depth_num;
	float interval;
	in >> cam.depth_min >> interval >> depth_num >> cam.depth_max;
	// ====================================================================
	////DTU version read
	// float depth_num = 192;
	// float interval;
	// in >> cam.depth_min >> interval;
	// cam.depth_max = interval * depth_num + cam.depth_min;
	////====================================================================
	in.close();
	return true;
}

bool ShowDepthMap(const path &depth_path, const cv::Mat& depth, float depth_min, float depth_max)
{
	const float deltaDepth = depth_max - depth_min;
	// save image
	cv::Mat result_img(depth.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < depth.cols; i++) {
		for (int j = 0; j < depth.rows; j++) {
			if (depth.at<float>(j, i) < depth_min || depth.at<float>(j, i) > depth_max || isnan(depth.at<float>(j, i))) {
				continue;
			}
			float pixel_val = (depth_max - depth.at<float>(j, i)) / deltaDepth;
			if (pixel_val > 1) {
				pixel_val = 1;
			}
			if (pixel_val < 0) {
				pixel_val = 0;
			}
			pixel_val = pixel_val * 255;
			if (pixel_val > 255) {
				pixel_val = 255;
			}
			else if (pixel_val< 0) {
				pixel_val = 0;
			}
			auto &pixel = result_img.at<cv::Vec3b>(j, i);
			if (pixel_val <= 51)
			{
				pixel[0] = 255;
				pixel[1] = pixel_val * 5;
				pixel[2] = 0;
			}
			else if (pixel_val <= 102)
			{
				pixel_val -= 51;
				pixel[0] = 255 - pixel_val * 5;
				pixel[1] = 255;
				pixel[2] = 0;
			}
			else if (pixel_val <= 153)
			{
				pixel_val -= 102;
				pixel[0] = 0;
				pixel[1] = 255;
				pixel[2] = pixel_val * 5;
			}
			else if (pixel_val <= 204)
			{
				pixel_val -= 153;
				pixel[0] = 0;
				pixel[1] = 255 - static_cast<unsigned char>(pixel_val * 128.0 / 51 + 0.5);
				pixel[2] = 255;
			}
			else if (pixel_val <= 255)
			{
				pixel_val -= 204;
				pixel[0] = 0;
				pixel[1] = 127 - static_cast<unsigned char>(pixel_val * 127.0 / 51 + 0.5);
				pixel[2] = 255;
			}
			
		}
	}
	cv::imwrite(depth_path.string(), result_img);
	return true;
}

bool ShowNormalMap(const path &normal_path, const cv::Mat &normal)
{
	if (normal.empty()) {
		return false;
	}
	cv::Mat normalized_normal = normal.clone();
	for (int i = 0; i < normalized_normal.rows; i++) {
		for (int j = 0; j < normalized_normal.cols; j++) {
			cv::Vec3f normal_val = normalized_normal.at<cv::Vec3f>(i, j);
			float norm = sqrt(pow(normal_val[0], 2) + pow(normal_val[1], 2) + pow(normal_val[2], 2));
			if (norm == 0) {
				normalized_normal.at<cv::Vec3f>(i, j) = cv::Vec3f(0, 0, 0);
			}
			else {
				normalized_normal.at<cv::Vec3f>(i, j) = normal_val / norm;
			}
		}
	}

	cv::Mat img(normalized_normal.size(), CV_8UC3, cv::Scalar(0.f, 0.f, 0.f));
	normalized_normal.convertTo(img, img.type(), 255.f / 2.f, 255.f / 2.f);
	cv::imwrite(normal_path.string(), img);
	return true;
}

bool ShowWeakImage(const path &weak_path, const cv::Mat &weak) {
	// show image
	if (weak.empty()) {
		return false;
	}
	const int width = weak.cols;
	const int height = weak.rows;
	cv::Mat weak_info_image(height, width, CV_8UC3);
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			switch (weak.at<uchar>(r, c))
			{
			case WEAK:
				weak_info_image.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
				break;
			case STRONG:
				weak_info_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 255, 0);
				break;
			case UNKNOWN:
				weak_info_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 255);
				break;
			}
		}
	}
	// save
	cv::imwrite(weak_path.string(), weak_info_image);
	return true;
}

bool ShowEdgeImage(const path &edge_path, const cv::Mat &edge) {
	if (edge.empty()) {
		return false;
	}
	// save new edge image
	int height = edge.rows;
	int width = edge.cols;
	cv::Mat edge_image(height, width, CV_8UC3);
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			switch (edge.at<uchar>(r, c))
			{
			case 0:
				edge_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
				break;
			case 255:
				edge_image.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
				break;
			default:
				edge_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 255);
				break;
			}
		}
	}
	cv::imwrite(edge_path.string(), edge_image);
	return true;
}

bool ExportPointCloud(const path& point_cloud_path, std::vector<PointList>& pointcloud)
{
	ofstream out(point_cloud_path, std::ios::binary);
	if (out.bad()) {
		return false;
	}

	out << "ply\n";
	out << "format binary_little_endian 1.0\n";
	out << "element vertex " << int(pointcloud.size()) << "\n";
	out << "property float x\n";
	out << "property float y\n";
	out << "property float z\n";
	out << "property uchar diffuse_blue\n";
	out << "property uchar diffuse_green\n";
	out << "property uchar diffuse_red\n";
	out << "end_header\n";

	for (size_t idx = 0; idx < pointcloud.size(); idx++)
	{
		float px = pointcloud[idx].coord.x;
		float py = pointcloud[idx].coord.y;
		float pz = pointcloud[idx].coord.z;


		cv::Vec3b pixel;
		pixel[0] = static_cast<uchar>(pointcloud[idx].color.x);
		pixel[1] = static_cast<uchar>(pointcloud[idx].color.y);
		pixel[2] = static_cast<uchar>(pointcloud[idx].color.z);

		out.write((char *)&px, sizeof(float));
		out.write((char *)&py, sizeof(float));
		out.write((char *)&pz, sizeof(float));

		out.write((char *)&pixel[0], sizeof(uchar));
		out.write((char *)&pixel[1], sizeof(uchar));
		out.write((char *)&pixel[2], sizeof(uchar));
	}
	out.close();
	return true;
}

bool ExportFusedPointCloud(const path& point_cloud_path, const std::vector<PointList>& pointcloud)
{
	ofstream out(point_cloud_path, std::ios::binary);
	if (!out) {
		return false;
	}
	out << "ply\nformat binary_little_endian 1.0\n";
	out << "element vertex " << pointcloud.size() << "\n";
	out << "property float x\nproperty float y\nproperty float z\n";
	out << "property float nx\nproperty float ny\nproperty float nz\n";
	out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
	out << "end_header\n";
	for (const auto& point : pointcloud) {
		// Write fields explicitly: do not serialize struct alignment/padding.
		const float values[6] = { point.coord.x, point.coord.y, point.coord.z,
			point.normal.x, point.normal.y, point.normal.z };
		out.write(reinterpret_cast<const char*>(values), sizeof(values));
		// OpenCV stores colors in BGR order.
		const unsigned char rgb[3] = { static_cast<unsigned char>(point.color.z),
			static_cast<unsigned char>(point.color.y), static_cast<unsigned char>(point.color.x) };
		out.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
	}
	out.close();
	return static_cast<bool>(out);
}

void StringAppendV(std::string* dst, const char* format, va_list ap) {
	// First try with a small fixed size buffer.
	static const int kFixedBufferSize = 1024;
	char fixed_buffer[kFixedBufferSize];

	// It is possible for methods that use a va_list to invalidate
	// the data in it upon use.  The fix is to make a copy
	// of the structure before using it and use that copy instead.
	va_list backup_ap;
	va_copy(backup_ap, ap);
	int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
	va_end(backup_ap);

	if (result < kFixedBufferSize) {
		if (result >= 0) {
			// Normal case - everything fits.
			dst->append(fixed_buffer, result);
			return;
		}

#ifdef _MSC_VER
		// Error or MSVC running out of space.  MSVC 8.0 and higher
		// can be asked about space needed with the special idiom below:
		va_copy(backup_ap, ap);
		result = vsnprintf(nullptr, 0, format, backup_ap);
		va_end(backup_ap);
#endif

		if (result < 0) {
			// Just an error.
			return;
		}
	}

	// Increase the buffer size to the size requested by vsnprintf,
	// plus one for the closing \0.
	const int variable_buffer_size = result + 1;
	std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);

	// Restore the va_list before we use it again.
	va_copy(backup_ap, ap);
	result =
		vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
	va_end(backup_ap);

	if (result >= 0 && result < variable_buffer_size) {
		dst->append(variable_buffer.get(), result);
	}
}

std::string StringPrintf(const char* format, ...) {
	va_list ap;
	va_start(ap, format);
	std::string result;
	StringAppendV(&result, format, ap);
	va_end(ap);
	return result;
}

void CudaSafeCall(const cudaError_t error, const std::string& file,
	const int line) {
	if (error != cudaSuccess) {
		std::cerr << StringPrintf("%s in %s at line %i", cudaGetErrorString(error),
			file.c_str(), line)
			<< std::endl;
		exit(EXIT_FAILURE);
	}
}

void CudaCheckError(const char* file, const int line) {
	cudaError error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cerr << StringPrintf("cudaCheckError() failed at %s:%i : %s", file,
			line, cudaGetErrorString(error))
			<< std::endl;
		exit(EXIT_FAILURE);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	error = cudaDeviceSynchronize();
	if (cudaSuccess != error) {
		std::cerr << StringPrintf("cudaCheckError() with sync failed at %s:%i : %s",
			file, line, cudaGetErrorString(error))
			<< std::endl;
		std::cerr
			<< "This error is likely caused by the graphics card timeout "
			"detection mechanism of your operating system. Please refer to "
			"the FAQ in the documentation on how to solve this problem."
			<< std::endl;
		exit(EXIT_FAILURE);
	}
}

std::string ToFormatIndex(int index) {
	std::stringstream ss;
	ss << std::setw(8) << std::setfill('0') << index;
	return ss.str();
}

DPE::DPE(const Problem &problem) {
	params_host = problem.params;
	this->problem = problem;
}

DPE::~DPE() {
	delete[] plane_hypotheses_host;

	if (problem.params.use_edge || problem.params.use_limit) {
		cudaFree(edge_cuda);
	}
	if (problem.params.use_edge) {
		cudaFree(edge_neigh_cuda);
		cudaFree(complex_cuda);
	}
	if (problem.params.use_label) {
		cudaFree(label_cuda);
		cudaFree(label_boundary_cuda);
	}
	if (problem.params.use_radius) {
		cudaFree(radius_cuda);
	}
	// free images
	{
		for (int i = 0; i < num_images; ++i) {
			cudaDestroyTextureObject(texture_objects_host.images[i]);
			cudaFreeArray(cuArray[i]);
		}
		cudaFree(texture_objects_cuda);
	}
	// may free depths
	if (params_host.geom_consistency) {
		for (int i = 0; i < num_images; ++i) {
			cudaDestroyTextureObject(texture_depths_host.images[i]);
			cudaFreeArray(cuDepthArray[i]);
		}
		cudaFree(texture_depths_cuda);
	}
	// may free supports
	cudaFree(cameras_cuda);
	cudaFree(plane_hypotheses_cuda);
	cudaFree(fit_plane_hypotheses_cuda);
	cudaFree(fit_plane_valid_cuda);
	cudaFree(costs_cuda);
	cudaFree(rand_states_cuda);
	cudaFree(selected_views_cuda);
	cudaFree(params_cuda);
	cudaFree(helper_cuda);
	cudaFree(neighbours_cuda);
	cudaFree(neigbours_map_cuda);
	cudaFree(weak_info_cuda);
	cudaFree(weak_reliable_cuda);
	cudaFree(view_weight_cuda);
	cudaFree(weak_nearest_strong);
	cudaFree(adaptive_refinement_mask_cuda);
	cudaFree(active_pixel_indices_cuda);
#ifdef DEBUG_COST_LINE
	cudaFree(weak_ncc_cost_cuda);
#endif // DEBUG_COST_LINE

}

void DPE::InuputInitialization() {
	images.clear();
	cameras.clear();
	// get folder
	path image_folder = problem.dense_folder / path("images");
	path cam_folder = problem.dense_folder / path("cams");
	//path weak_folder = problem.dense_folder / path("weaks");
	// =================================================
	// read ref image and src images
	// ref
	{ 
		path ref_image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		cv::Mat_<uint8_t> image_uint = cv::imread(ref_image_path.string(), cv::IMREAD_GRAYSCALE);
		cv::Mat image_float;
		image_uint.convertTo(image_float, CV_32FC1);
		images.push_back(image_float);
		width = image_float.cols;
		height = image_float.rows;
	}
	// src
	for (const auto &src_idx : problem.src_image_ids) {
		path src_image_path = image_folder / path(ToFormatIndex(src_idx) + ".jpg");
		cv::Mat_<uint8_t> image_uint = cv::imread(src_image_path.string(), cv::IMREAD_GRAYSCALE);
		cv::Mat image_float;
		image_uint.convertTo(image_float, CV_32FC1);
		images.push_back(image_float);
		// assert: images_float.cols == width;
		// assert: images_float.rows == height;
	}
	if (images.size() > MAX_IMAGES) {
		std::cerr << "Can't process so much images: " << images.size() << std::endl;
		exit(EXIT_FAILURE);
	}
	// =================================================
	// read ref camera and src camera
	// ref
	{
		path ref_cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		Camera cam;
		ReadCamera(ref_cam_path, cam);
		cam.width = width;
		cam.height = height;
		cameras.push_back(cam);
	}
	// src
	for (const auto &src_idx : problem.src_image_ids) {
		path src_cam_path = cam_folder / path(ToFormatIndex(src_idx) + "_cam.txt");
		Camera cam;
		ReadCamera(src_cam_path, cam);
		cam.width = width;
		cam.height = height;
		cameras.push_back(cam);
	}
	// =================================================
	// set some params
	params_host.depth_min = cameras[0].depth_min * 0.6f;
	params_host.depth_max = cameras[0].depth_max * 1.2f;
	params_host.num_images = (int)images.size();
	num_images = (int)images.size();
	// =================================================
	std::cout << "Read images and camera done\n";
	std::cout << "Depth range: " << params_host.depth_min << " " << params_host.depth_max << std::endl;
	std::cout << "Num images: " << params_host.num_images << std::endl;
	// =================================================
	// scale images
    // Limit the base image before the existing power-of-two pyramid.
    for (int i = 0; i < num_images; ++i) {
        const cv::Size target = LimitedImageSize(images[i].cols, images[i].rows, problem.params.max_image_size);
        const float sx = target.width / static_cast<float>(images[i].cols);
        const float sy = target.height / static_cast<float>(images[i].rows);
        if (target != images[i].size()) {
            cv::resize(images[i], images[i], target, 0, 0, cv::INTER_LINEAR);
            cameras[i].K[0] *= sx; cameras[i].K[2] *= sx;
            cameras[i].K[4] *= sy; cameras[i].K[5] *= sy;
        }
        cameras[i].width = target.width; cameras[i].height = target.height;
    }
    width = images[0].cols; height = images[0].rows;
	if (problem.scale_size != 1) {
		for (int i = 0; i < num_images; ++i) {
			const float factor = 1.0f / (float)(problem.scale_size);
			const int new_cols = std::round(images[i].cols * factor);
			const int new_rows = std::round(images[i].rows * factor);

			const float scale_x = new_cols / static_cast<float>(images[i].cols);
			const float scale_y = new_rows / static_cast<float>(images[i].rows);

			cv::Mat_<float> scaled_image_float;
			cv::resize(images[i], scaled_image_float, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
			images[i] = scaled_image_float.clone();

			width = scaled_image_float.cols;
			height = scaled_image_float.rows;

			cameras[i].K[0] *= scale_x;
			cameras[i].K[2] *= scale_x;
			cameras[i].K[4] *= scale_y;
			cameras[i].K[5] *= scale_y;
			cameras[i].width = width;
			cameras[i].height = height;
		}
		std::cout << "Scale images and cameras done\n";
	}
	std::cout << "Image size: " << width << " * " << height << std::endl;
	// =================================================
	// read depth form geom consistency
	if (params_host.geom_consistency) {
		depths.clear();
		path ref_depth_path = problem.result_folder / path("depths.dmb");
		cv::Mat ref_depth;
		ReadBinMat(ref_depth_path, ref_depth);
		depths.push_back(ref_depth);
		for (const auto &src_idx : problem.src_image_ids) {
			path src_depth_path = problem.dense_folder / path("DPE") / path(ToFormatIndex(src_idx)) / path("depths.dmb");
			cv::Mat src_depth;
			ReadBinMat(src_depth_path, src_depth);
			depths.push_back(src_depth);
		}
		for (auto &depth : depths) {
			if (depth.cols != width || depth.rows != height) {
				RescaleMatToTargetSize<float>(depth, depth, cv::Size(width, height));
			}
		}

	}
	// =================================================
	// read weak info
	if (params_host.use_APD) {
		path weak_info_path = problem.result_folder / path("weak.bin");
		if (!exists(weak_info_path)) {
			std::cerr << "Can't find weak info file: " << weak_info_path.string() << std::endl;
			exit(EXIT_FAILURE);
		}
		ReadBinMat(weak_info_path, weak_info_host);
		if (weak_info_host.cols != width || weak_info_host.rows != height) {
			std::cerr << "Weak info doesn't match the images' size!\n";
			RescaleMatToTargetSize<uchar>(weak_info_host, weak_info_host, cv::Size(width, height));
			std::cout << "Scale done\n";
		}
		
		neighbours_map_host = cv::Mat::zeros(weak_info_host.size(), CV_32SC1);
		weak_count = 0;
		for (int r = 0; r < weak_info_host.rows; ++r) {
			for (int c = 0; c < weak_info_host.cols; ++c) {
				int val = weak_info_host.at<uchar>(r, c);
				// point is not strong
				if (val == WEAK) {
					neighbours_map_host.at<int>(r, c) = weak_count;
					weak_count++;
				}
			}
		}
		std::cout << "Weak count: " << weak_count << " / " << weak_info_host.cols * weak_info_host.rows << " = " << (float)weak_count / (float)(weak_info_host.cols * weak_info_host.rows) * 100 << "%" << std::endl;
	}
	else {
		weak_info_host = cv::Mat::zeros(height, width, CV_8UC1);
		weak_count = 0;
		for (int r = 0; r < weak_info_host.rows; ++r) {
			for (int c = 0; c < weak_info_host.cols; ++c) {
				weak_info_host.at<uchar>(r, c) = STRONG;
			}
		}
	}
	// =================================================
	plane_hypotheses_host = new float4[cameras[0].height * cameras[0].width];
	selected_views_host = cv::Mat::zeros(height, width, CV_32SC1);
	if (params_host.state != FIRST_INIT) {
		// input plane hypotheses from existed result
		path depth_path = problem.result_folder / path("depths.dmb");
		path normal_path = problem.result_folder / path("normals.dmb");
		cv::Mat depth, normal;
		ReadBinMat(depth_path, depth);
		ReadBinMat(normal_path, normal);
		if (depth.cols != width || depth.rows != height || normal.cols != width || normal.rows != height) {
			std::cerr << "Depth and Normal doesn't match the images' size!\n";
			RescaleMatToTargetSize<float>(depth, depth, cv::Size2i(width, height));
			RescaleMatToTargetSize<cv::Vec3f>(normal, normal, cv::Size2i(width, height));
		}
		for (int col = 0; col < width; ++col) {
			for (int row = 0; row < height; ++row) {
				int center = row * width + col;
				plane_hypotheses_host[center].w = depth.at<float>(row, col);
				plane_hypotheses_host[center].x = normal.at<cv::Vec3f>(row, col)[0];
				plane_hypotheses_host[center].y = normal.at<cv::Vec3f>(row, col)[1];
				plane_hypotheses_host[center].z = normal.at<cv::Vec3f>(row, col)[2];
			}
		}
		{
			path selected_view_path = problem.result_folder / path("selected_views.bin");
			ReadBinMat(selected_view_path, selected_views_host);
			if (selected_views_host.cols != width || selected_views_host.rows != height) {
				std::cerr << "Select view doesn't match the images' size!\n";
				RescaleMatToTargetSize<unsigned int>(selected_views_host, selected_views_host, cv::Size2i(width, height));
			}
		}
	}
	// =================================================
}

void DPE::CudaSpaceInitialization() {
	// =================================================
	// move images to gpu
	for (int i = 0; i < num_images; ++i) {
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cudaMallocArray(&cuArray[i], &channelDesc, width, height);
		cudaMemcpy2DToArray(cuArray[i], 0, 0, images[i].ptr<float>(), images[i].step[0], width * sizeof(float), height, cudaMemcpyHostToDevice);
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray[i];
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;
		cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
	}
	cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
	cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);
	// may move depths to gpu
	if (params_host.geom_consistency) {
		for (int i = 0; i < num_images; ++i) {
			int height = depths[i].rows;
			int width = depths[i].cols;
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
			cudaMallocArray(&cuDepthArray[i], &channelDesc, width, height);
			cudaMemcpy2DToArray(cuDepthArray[i], 0, 0, depths[i].ptr<float>(), depths[i].step[0], width * sizeof(float), height, cudaMemcpyHostToDevice);
			struct cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(cudaResourceDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = cuDepthArray[i];
			struct cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(cudaTextureDesc));
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = 0;
			cudaCreateTextureObject(&(texture_depths_host.images[i]), &resDesc, &texDesc, NULL);
		}
		cudaMalloc((void**)&texture_depths_cuda, sizeof(cudaTextureObjects));
		cudaMemcpy(texture_depths_cuda, &texture_depths_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);
	}
	// =================================================
	// move camera to gpu
	cudaMalloc((void**)&cameras_cuda, sizeof(Camera) * (num_images));
	cudaMemcpy(cameras_cuda, &cameras[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);
	// malloc memory for important data structure
	const int length = width * height;
	// define cost
	cudaMalloc((void**)&costs_cuda, sizeof(float) * length);
	// malloc memory for rand states
	cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * length);
	// malloc for selected_views
	cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * length);
	cudaMemcpy(selected_views_cuda, selected_views_host.ptr<unsigned int>(0), sizeof(unsigned int) * length, cudaMemcpyHostToDevice);
	// view weight
	cudaMalloc((void**)&view_weight_cuda, sizeof(uchar) * length * MAX_IMAGES);
	// move plane hypotheses to gpu
	cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * length);
	cudaMemcpy(plane_hypotheses_cuda, plane_hypotheses_host, sizeof(float4) * length, cudaMemcpyHostToDevice);
	// malloc memory for fit plane 
	cudaMalloc((void**)&fit_plane_hypotheses_cuda, sizeof(float4) * length);
	cudaMemset(fit_plane_hypotheses_cuda, 0, sizeof(float4) * length);
	cudaMalloc((void**)&fit_plane_valid_cuda, length);
	cudaMemset(fit_plane_valid_cuda, 0, length);

	// malloc edge array
	if (problem.params.use_edge || problem.params.use_limit) {
		cudaMalloc((void**)&edge_cuda, sizeof(uint8_t) * length);
		cudaMemcpy(edge_cuda, edge_host.ptr<uchar>(0), sizeof(uchar) * length, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&edge_low_res_cuda, sizeof(uint8_t) * low_height * low_width);
		cudaMemcpy(edge_low_res_cuda, edge_low_res_host.ptr<uchar>(0), sizeof(uchar) * low_height * low_width, cudaMemcpyHostToDevice);
	}
	if (problem.params.use_edge) {
		cudaMalloc((void **)(&edge_neigh_cuda), length * 8 * sizeof(short2));
		cudaMalloc((void**)&complex_cuda, sizeof(float) * length);
	}
	if (problem.params.use_label) {
		cudaMalloc((void**)(&label_cuda), length * sizeof(int));
		cudaMemcpy(label_cuda, label_host.ptr<int>(0), sizeof(int) * (height * width), cudaMemcpyHostToDevice);
		cudaMalloc((void **)(&label_boundary_cuda), weak_count * 8 * sizeof(short2));
	}
	if (problem.params.use_radius) {
		cudaMalloc((void**)(&radius_cuda), length * sizeof(int));
	}
	if (params_host.adaptive_refinement) {
		cudaMalloc((void**)(&adaptive_refinement_mask_cuda), length * sizeof(uchar));
		cudaMemcpy(adaptive_refinement_mask_cuda, adaptive_refinement_mask_host.ptr<uchar>(0),
			length * sizeof(uchar), cudaMemcpyHostToDevice);
		active_pixel_count = length;
		if (adaptive_frozen_fraction >= 0.25f &&
			cv::countNonZero(adaptive_refinement_mask_host) < length) {
			std::vector<int> active_pixels;
			active_pixels.reserve(static_cast<size_t>(length * (1.0f - adaptive_frozen_fraction)));
			for (int r = 0; r < height; ++r) {
				const uchar *mask_row = adaptive_refinement_mask_host.ptr<uchar>(r);
				for (int c = 0; c < width; ++c)
					if (mask_row[c] != 0) active_pixels.push_back(r * width + c);
			}
			active_pixel_count = static_cast<int>(active_pixels.size());
			if (active_pixel_count > 0) {
				cudaMalloc((void**)(&active_pixel_indices_cuda), active_pixel_count * sizeof(int));
				cudaMemcpy(active_pixel_indices_cuda, active_pixels.data(), active_pixel_count * sizeof(int),
					cudaMemcpyHostToDevice);
			}
		}
	}

	// malloc memory for weak info
	cudaMalloc((void **)(&weak_info_cuda), length * sizeof(uchar));
	cudaMemcpy(weak_info_cuda, weak_info_host.ptr<uchar>(0), length * sizeof(uchar), cudaMemcpyHostToDevice);
	// malloc memory for weak reliable info
	cudaMalloc((void **)(&weak_reliable_cuda), length * sizeof(uchar));
	// malloc memory for nearest strong points
	cudaMalloc((void**)(&weak_nearest_strong), length * sizeof(short2));
	// move neighbour map to gpu
	cudaMalloc((void**)(&neigbours_map_cuda), length * sizeof(int));
	cudaMemcpy(neigbours_map_cuda, neighbours_map_host.ptr<int>(0), length * sizeof(int), cudaMemcpyHostToDevice);
	// malloc memory for deformable ncc
	cudaMalloc((void **)(&neighbours_cuda), weak_count * NEIGHBOUR_NUM * sizeof(short2));
	// move param to gpu
	cudaMalloc((void**)(&params_cuda), sizeof(PatchMatchParams));
	cudaMemcpy(params_cuda, &params_host, sizeof(PatchMatchParams), cudaMemcpyHostToDevice);
	// =================================================
#ifdef DEBUG_COST_LINE
	cudaMalloc((void**)(&weak_ncc_cost_cuda), sizeof(float) * width * height * 61);
#endif // DEBUG_COST_LINE
}

void DPE::SupportInitialization() {
	int scale = 0;
	while ((1 << scale) < problem.scale_size) scale++;

	if (problem.params.use_edge || problem.params.use_limit) {
		int scale = 0;
		while((1 << scale) < problem.scale_size) scale++;
		path edge_path = problem.result_folder / path("edges_max" + std::to_string(problem.params.max_image_size) + "_" + std::to_string(scale) + ".dmb");
		// read image edge info
		ReadBinMat(edge_path, edge_host);

		int max_scale = 0;
		if (problem.params.high_res_img) {
			while((1 << max_scale) < problem.params.max_scale_size) max_scale++;
		} else {
			max_scale = scale;
		}
		path edge_low_res_path = problem.result_folder / path("edges_max" + std::to_string(problem.params.max_image_size) + "_" + std::to_string(max_scale) + ".dmb");
		ReadBinMat(edge_low_res_path, edge_low_res_host);
		low_width = edge_low_res_host.cols;
		low_height = edge_low_res_host.rows;
	}

	if (problem.params.use_label) {
		path label_path = problem.result_folder / path("labels_max" + std::to_string(problem.params.max_image_size) + "_" + std::to_string(scale) + ".dmb");
		ReadBinMat(label_path, label_host);
	}

	if (!problem.params.adaptive_refinement) return;
	const bool profile = std::getenv("DPE_PROFILE") != nullptr;
	auto profile_start = std::chrono::steady_clock::now();
	const auto profile_stage = [&](const char *name) {
		if (!profile) return;
		const auto now = std::chrono::steady_clock::now();
		std::cout << "Profile freeze " << name << ": "
			<< std::chrono::duration<double, std::milli>(now - profile_start).count() << " ms\n";
		profile_start = now;
	};
	adaptive_refinement_mask_host = cv::Mat(height, width, CV_8UC1, cv::Scalar(1));
	cv::Mat adaptive_stability_count_host = cv::Mat::zeros(height, width, CV_8UC1);
	const path adaptive_mask_path = problem.result_folder / path("adaptive_frozen.dmb");
	const path adaptive_stability_path = problem.result_folder / path("adaptive_stability.dmb");
	auto &adaptive_state = *problem.adaptive_state;
	if (problem.params.state == FIRST_INIT) adaptive_state = AdaptiveRefinementState();
	if (problem.params.state != FIRST_INIT) {
		cv::Mat previous_mask = adaptive_state.mask;
		if (previous_mask.empty() && exists(adaptive_mask_path))
			ReadBinMat(adaptive_mask_path, previous_mask);
		if (!previous_mask.empty() && previous_mask.type() == CV_8UC1) {
			if (previous_mask.size() != adaptive_refinement_mask_host.size()) {
				const int previous_frozen = previous_mask.total() - cv::countNonZero(previous_mask);
				adaptive_refinement_mask_host = PropagateFrozenMask(previous_mask, adaptive_refinement_mask_host.size());
				const int inherited_frozen = adaptive_refinement_mask_host.total() -
					cv::countNonZero(adaptive_refinement_mask_host);
				std::cout << "Adaptive freeze propagation: " << previous_mask.cols << "x" << previous_mask.rows
					<< " (" << previous_frozen << " frozen) -> " << adaptive_refinement_mask_host.cols << "x"
					<< adaptive_refinement_mask_host.rows << " (" << inherited_frozen
					<< " frozen; one representative per frozen parent)\n";
			} else {
				adaptive_refinement_mask_host = previous_mask;
			}
		}
	}
	if (problem.params.state != FIRST_INIT) {
		cv::Mat previous_stability_count = adaptive_state.stability;
		if (previous_stability_count.empty() && exists(adaptive_stability_path))
			ReadBinMat(adaptive_stability_path, previous_stability_count);
		if (!previous_stability_count.empty() &&
			previous_stability_count.type() == CV_8UC1) {
			if (previous_stability_count.size() != adaptive_stability_count_host.size())
				previous_stability_count = PropagateStabilityCount(previous_stability_count,
					adaptive_stability_count_host.size());
			adaptive_stability_count_host = previous_stability_count;
		}
	}
	profile_stage("load state");
	adaptive_state.mask = adaptive_refinement_mask_host;
	adaptive_state.stability = adaptive_stability_count_host;
	adaptive_frozen_fraction = 1.0f - static_cast<float>(cv::countNonZero(adaptive_refinement_mask_host)) /
		static_cast<float>(std::max(1, width * height));
	profile_stage("cache state");
}

void DPE::SetDataPassHelperInCuda() {
	helper_host.width = this->width;
	helper_host.height = this->height;
	helper_host.low_width = this->low_width;
	helper_host.low_height = this->low_height;
	helper_host.ref_index = this->problem.ref_image_id;
	helper_host.texture_depths_cuda = this->texture_depths_cuda;
	helper_host.texture_objects_cuda = this->texture_objects_cuda;
	helper_host.cameras_cuda = this->cameras_cuda;
	helper_host.costs_cuda = this->costs_cuda;
	helper_host.neighbours_cuda = this->neighbours_cuda;
	helper_host.neighbours_map_cuda = this->neigbours_map_cuda;
	helper_host.plane_hypotheses_cuda = this->plane_hypotheses_cuda;
	helper_host.rand_states_cuda = this->rand_states_cuda;
	helper_host.selected_views_cuda = this->selected_views_cuda;
	helper_host.weak_info_cuda = this->weak_info_cuda;
	helper_host.params = params_cuda;
	helper_host.debug_point = make_int2(DEBUG_POINT_X, DEBUG_POINT_Y);
	helper_host.show_ncc_info = false;
	helper_host.fit_plane_hypotheses_cuda = fit_plane_hypotheses_cuda;
	helper_host.fit_plane_valid_cuda = fit_plane_valid_cuda;
	helper_host.weak_reliable_cuda = weak_reliable_cuda;
	helper_host.view_weight_cuda = view_weight_cuda;
	helper_host.weak_nearest_strong = weak_nearest_strong;
	helper_host.edge_cuda = edge_cuda;
	helper_host.edge_low_res_cuda = edge_low_res_cuda;
	helper_host.edge_neigh_cuda = edge_neigh_cuda;
	helper_host.label_cuda = label_cuda;
	helper_host.label_boundary_cuda = label_boundary_cuda;
	helper_host.complex_cuda = complex_cuda;
	helper_host.radius_cuda = radius_cuda;
	helper_host.adaptive_refinement_mask_cuda = adaptive_refinement_mask_cuda;
	helper_host.active_pixel_indices_cuda = active_pixel_indices_cuda;
	helper_host.active_pixel_count = active_pixel_count;
#ifdef DEBUG_COST_LINE
	helper_host.weak_ncc_cost_cuda = weak_ncc_cost_cuda;
#endif // DEBUG_COST_LINE
	cudaMalloc((void**)(&helper_cuda), sizeof(DataPassHelper));
	cudaMemcpy(helper_cuda, &helper_host, sizeof(DataPassHelper), cudaMemcpyHostToDevice);
}

float4 DPE::GetPlaneHypothesis(int r, int c) {
	return plane_hypotheses_host[c + r * width];
}

cv::Mat DPE::GetEdge() {
	return edge_host;
}

cv::Mat DPE::GetPixelStates() {
	return weak_info_host;
}

cv::Mat DPE::GetConfidenceCosts() {
	return confidence_cost_host;
}

void DPE::UpdateAdaptiveMaskFromConfidence(const cv::Mat &pixel_states,
	const cv::Mat &confidence_costs) {
	if (!problem.params.adaptive_refinement || pixel_states.empty() ||
		pixel_states.size() != adaptive_refinement_mask_host.size() ||
		confidence_costs.empty() || confidence_costs.size() != pixel_states.size() ||
		confidence_costs.type() != CV_32FC1) return;
	AdaptiveRefinementState &state = *problem.adaptive_state;
	constexpr float kWeakFreezeCostThreshold = 0.15f;
	int newly_frozen = 0;
	int newly_frozen_strong = 0;
	int newly_frozen_weak_low_cost = 0;
	for (int r = 0; r < pixel_states.rows; ++r) {
		const uchar *state_row = pixel_states.ptr<uchar>(r);
		const float *cost_row = confidence_costs.ptr<float>(r);
		uchar *mask_row = adaptive_refinement_mask_host.ptr<uchar>(r);
		uchar *stability_row = state.stability.ptr<uchar>(r);
		for (int c = 0; c < pixel_states.cols; ++c) {
			const bool strong = state_row[c] == STRONG;
			const bool weak_low_cost = state_row[c] == WEAK &&
				IsFiniteFloat(cost_row[c]) && cost_row[c] <= kWeakFreezeCostThreshold;
			if (mask_row[c] != 0 && (strong || weak_low_cost)) {
				mask_row[c] = 0;
				stability_row[c] = 1;
				++newly_frozen;
				if (strong) ++newly_frozen_strong;
				else ++newly_frozen_weak_low_cost;
			}
		}
	}
	state.mask = adaptive_refinement_mask_host;
	adaptive_frozen_fraction = 1.0f - static_cast<float>(cv::countNonZero(adaptive_refinement_mask_host)) /
		static_cast<float>(std::max(1, adaptive_refinement_mask_host.rows * adaptive_refinement_mask_host.cols));
	std::cout << "Adaptive confidence freeze: " << newly_frozen << " newly frozen ("
		<< newly_frozen_strong << " strong, " << newly_frozen_weak_low_cost
		<< " weak cost <= " << kWeakFreezeCostThreshold << "), "
		<< (100.0f * adaptive_frozen_fraction) << "% total" << std::endl;
}

cv::Mat DPE::GetSelectedViews() {
	return selected_views_host;
}

float DPE::GetAdaptiveFrozenFraction() const {
	return adaptive_frozen_fraction;
}

int DPE::GetWidth() {
	return width;
}

int DPE::GetHeight() {
	return height;
}

float DPE::GetDepthMin() {
	return params_host.depth_min;
}

float DPE::GetDepthMax() {
	return params_host.depth_max;
}

void RescaleImageAndCamera(cv::Mat &src, cv::Mat &dst, cv::Mat &depth, Camera &camera)
{
	const int cols = depth.cols;
	const int rows = depth.rows;

	if (cols == src.cols && rows == src.rows) {
		dst = src.clone();
		return;
	}

	const float scale_x = cols / static_cast<float>(src.cols);
	const float scale_y = rows / static_cast<float>(src.rows);

	cv::resize(src, dst, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);

	camera.K[0] *= scale_x;
	camera.K[2] *= scale_x;
	camera.K[4] *= scale_y;
	camera.K[5] *= scale_y;
	camera.width = cols;
	camera.height = rows;
}

template <typename TYPE>
void RescaleMatToTargetSize(const cv::Mat &src, cv::Mat &dst, const cv::Size2i &target_size) {
	if (src.cols == target_size.width && src.rows == target_size.height) {
		return;
	}
	const float scale_x = target_size.width / static_cast<float>(src.cols);
	const float scale_y = target_size.height / static_cast<float>(src.rows);

	int type = src.type();
	cv::Mat src_clone = src.clone();
	dst = cv::Mat(target_size.height, target_size.width, type);

	for (int r = 0; r < target_size.height; ++r) {
		for (int c = 0; c < target_size.width; ++c) {
			int o_r = static_cast<int>(r / scale_x);
			int o_c = static_cast<int>(c / scale_y);
			if (o_r < 0 || o_c < 0 || o_r >= src_clone.rows || o_c >= src_clone.cols) {
				continue;
			}
			dst.at<TYPE>(r, c) = src_clone.at<TYPE>(o_r, o_c);
		}
	}
}

float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera)
{
	float3 pointX;
	float3 tmpX;
	// Reprojection
	pointX.x = depth * (x - camera.K[2]) / camera.K[0];
	pointX.y = depth * (y - camera.K[5]) / camera.K[4];
	pointX.z = depth;

	// Rotation
	tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
	tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
	tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

	// Transformation
	float3 C;
	C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
	C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
	C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
	pointX.x = tmpX.x + C.x;
	pointX.y = tmpX.y + C.y;
	pointX.z = tmpX.z + C.z;

	return pointX;
}

void ProjectCamera(const float3 PointX, const Camera camera, float2 &point, float &depth)
{
	float3 tmp;
	tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
	tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
	tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

	depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
	point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
	point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2)
{
	float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
	float angle = acosf(dot_product);
	//if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
	if (angle != angle)
		return 0.0f;

	return angle;
}

namespace {
struct PlanePatch {
	cv::Vec3f normal;
	float offset;
	int min_x;
	int min_y;
	int max_x;
	int max_y;
	int area;
	int sample_pixels[5];
};

struct PlanePatchMap {
	cv::Mat labels;
	std::vector<PlanePatch> patches;
};

cv::Vec3f WorldCameraCenter(const Camera &camera) {
	return cv::Vec3f(
		-(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]),
		-(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]),
		-(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]));
}

cv::Vec3f WorldRayDirection(const Camera &camera, int x, int y) {
	const float ray_x = (x - camera.K[2]) / camera.K[0];
	const float ray_y = (y - camera.K[5]) / camera.K[4];
	return cv::Vec3f(
		camera.R[0] * ray_x + camera.R[3] * ray_y + camera.R[6],
		camera.R[1] * ray_x + camera.R[4] * ray_y + camera.R[7],
		camera.R[2] * ray_x + camera.R[5] * ray_y + camera.R[8]);
}

bool IntersectPlaneAtPixel(const PlanePatch &patch, const Camera &camera,
		int x, int y, float &depth, cv::Vec3f *world_point = nullptr) {
	const cv::Vec3f center = WorldCameraCenter(camera);
	const cv::Vec3f ray = WorldRayDirection(camera, x, y);
	const float denominator = patch.normal.dot(ray);
	if (!IsFiniteFloat(denominator) || std::fabs(denominator) < 1e-8f) return false;
	depth = -(patch.normal.dot(center) + patch.offset) / denominator;
	if (!IsFiniteFloat(depth) || depth <= 0.0f) return false;
	if (world_point) *world_point = center + ray * depth;
	return true;
}

PlanePatchMap BuildPlanePatchMap(const cv::Mat &depth, const cv::Mat &normal,
		const Camera &camera) {
	PlanePatchMap result;
	result.labels = cv::Mat(depth.size(), CV_32SC1, cv::Scalar(-1));
	cv::Mat visited(depth.size(), CV_8UC1, cv::Scalar(0));
	const float normal_cosine = std::cos(8.0f * static_cast<float>(M_PI) / 180.0f);
	const int dx[4] = {1, -1, 0, 0};
	const int dy[4] = {0, 0, 1, -1};
	std::vector<int> region;
	std::vector<int> stack;
	region.reserve(1024);
	stack.reserve(1024);
	uint64_t planar_pixels = 0;

	for (int y = 0; y < depth.rows; ++y) {
		for (int x = 0; x < depth.cols; ++x) {
			if (visited.at<uchar>(y, x)) continue;
			visited.at<uchar>(y, x) = 1;
			const float seed_depth = depth.at<float>(y, x);
			const cv::Vec3f seed_normal_raw = normal.at<cv::Vec3f>(y, x);
			if (!IsFiniteFloat(seed_depth) || seed_depth <= 0.0f || !IsValidNormal(seed_normal_raw)) continue;
			const cv::Vec3f seed_normal = seed_normal_raw * (1.0f / std::sqrt(seed_normal_raw.dot(seed_normal_raw)));
			const float3 seed_world = Get3DPointonWorld(x, y, seed_depth, camera);
			const cv::Vec3f seed_point(seed_world.x, seed_world.y, seed_world.z);
			const float seed_offset = -seed_normal.dot(seed_point);
			region.clear();
			stack.clear();
			const int seed_index = y * depth.cols + x;
			stack.push_back(seed_index);
			visited.at<uchar>(y, x) = 2;
			int min_x = x, max_x = x, min_y = y, max_y = y;
			int min_x_index = seed_index, max_x_index = seed_index;
			int min_y_index = seed_index, max_y_index = seed_index;
			cv::Vec3f normal_sum(0, 0, 0), point_sum(0, 0, 0);

			while (!stack.empty()) {
				const int index = stack.back();
				stack.pop_back();
				const int cy = index / depth.cols;
				const int cx = index - cy * depth.cols;
				region.push_back(index);
				if (cx < min_x) { min_x = cx; min_x_index = index; }
				if (cx > max_x) { max_x = cx; max_x_index = index; }
				if (cy < min_y) { min_y = cy; min_y_index = index; }
				if (cy > max_y) { max_y = cy; max_y_index = index; }
				const cv::Vec3f nraw = normal.at<cv::Vec3f>(cy, cx);
				const cv::Vec3f n = nraw * (1.0f / std::sqrt(nraw.dot(nraw)));
				const float z = depth.at<float>(cy, cx);
				const float3 world3 = Get3DPointonWorld(cx, cy, z, camera);
				const cv::Vec3f point(world3.x, world3.y, world3.z);
				normal_sum += n;
				point_sum += point;

				for (int k = 0; k < 4; ++k) {
					const int nx = cx + dx[k], ny = cy + dy[k];
					if (nx < 0 || ny < 0 || nx >= depth.cols || ny >= depth.rows ||
						visited.at<uchar>(ny, nx)) continue;
					const float nz = depth.at<float>(ny, nx);
					const cv::Vec3f nnraw = normal.at<cv::Vec3f>(ny, nx);
					if (!IsFiniteFloat(nz) || nz <= 0.0f || !IsValidNormal(nnraw)) continue;
					const cv::Vec3f nn = nnraw * (1.0f / std::sqrt(nnraw.dot(nnraw)));
					if (std::fabs(seed_normal.dot(nn)) < normal_cosine) continue;
					const float3 neighbor3 = Get3DPointonWorld(nx, ny, nz, camera);
					const cv::Vec3f neighbor(neighbor3.x, neighbor3.y, neighbor3.z);
					const float tolerance = std::max(0.005f, 0.002f * seed_depth);
					if (std::fabs(seed_normal.dot(neighbor) + seed_offset) > tolerance) continue;
					visited.at<uchar>(ny, nx) = 2;
					stack.push_back(ny * depth.cols + nx);
				}
			}

			// Very small coplanar groups are cheaper and safer as individual samples.
			if (region.size() < 16) {
				for (int index : region) visited.at<uchar>(index / depth.cols, index % depth.cols) = 1;
				continue;
			}
		cv::Vec3f patch_normal = normal_sum * (1.0f / std::sqrt(normal_sum.dot(normal_sum)));
		const cv::Vec3f centroid = point_sum * (1.0f / static_cast<float>(region.size()));
		PlanePatch patch;
		patch.normal = patch_normal;
		patch.offset = -patch_normal.dot(centroid);
		patch.min_x = min_x; patch.max_x = max_x;
		patch.min_y = min_y; patch.max_y = max_y;
		patch.area = static_cast<int>(region.size());
		patch.sample_pixels[0] = seed_index;
		patch.sample_pixels[1] = min_x_index;
		patch.sample_pixels[2] = max_x_index;
		patch.sample_pixels[3] = min_y_index;
		patch.sample_pixels[4] = max_y_index;
		const int patch_id = static_cast<int>(result.patches.size());
		result.patches.push_back(patch);
		for (int index : region)
			result.labels.at<int>(index / depth.cols, index % depth.cols) = patch_id;
		planar_pixels += region.size();
		}
	}
	std::cout << "Plane patches: " << result.patches.size() << ", " << planar_pixels
		<< " pixels in patches / " << (depth.total()) << " total\n";
	return result;
}

}

// ETH version
void RunFusion(const path &dense_folder, const std::vector<Problem> &problems, int simple_region_stride,
		bool plane_fusion, int plane_sample_stride)
{
	if (simple_region_stride < 1) throw std::invalid_argument("simple_region_stride must be >= 1");
	if (plane_sample_stride < 1) throw std::invalid_argument("plane_sample_stride must be >= 1");
	int num_images = problems.size();
	path image_folder = dense_folder / path("images");
	path cam_folder = dense_folder / path("cams");

	std::vector<cv::Mat> images;
	std::vector<Camera> cameras;
	std::vector<cv::Mat> depths;
	std::vector<cv::Mat> normals;
	std::vector<cv::Mat> masks;
	std::vector<cv::Mat> blocks;
	std::vector<cv::Mat> weaks;
	images.clear();
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	blocks.clear();
	weaks.clear();
	std::unordered_map<int, int> imageIdToindexMap;

	path block_folder = dense_folder / path("blocks");
	bool use_block = false;
	if (exists(block_folder)) {
		use_block = true;
	}

	for (int i = 0; i < num_images; ++i) {
		const auto &problem = problems[i];
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		imageIdToindexMap.emplace(problem.ref_image_id, i);
		cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
		path cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		Camera camera;
		ReadCamera(cam_path, camera);
	
		path depth_path = problem.result_folder / path("depths.dmb");
		path normal_path = problem.result_folder / path("normals.dmb");
		path weak_path = problem.result_folder / path("weak.bin");
		cv::Mat depth, normal, weak;
		ReadBinMat(depth_path, depth);
		ReadBinMat(normal_path, normal);
		ReadBinMat(weak_path, weak);
	
		if (use_block) {
			path block_path = block_folder / path("mask_" + std::to_string(problem.ref_image_id) + ".jpg");
			cv::Mat block_jpg = cv::imread(block_path.string(), cv::IMREAD_GRAYSCALE);
			if (block_jpg.empty()) throw std::runtime_error("Cannot read mask: " + block_path.string());
			cv::resize(block_jpg, block_jpg, depth.size(), 0, 0, cv::INTER_NEAREST);
			blocks.emplace_back(block_jpg);
		}
	
		cv::Mat scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		if ((plane_fusion || simple_region_stride > 1) && normal.size() != depth.size())
			cv::resize(normal, normal, depth.size(), 0, 0, cv::INTER_NEAREST);
		images.emplace_back(scaled_image);
		cameras.emplace_back(camera);
		depths.emplace_back(depth);
		normals.emplace_back(normal);
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
		masks.emplace_back(mask);
		RescaleMatToTargetSize<uchar>(weak, weak, cv::Size2i(depth.cols, depth.rows));
		weaks.emplace_back(weak);
	}
	if (plane_fusion) {
		if (plane_sample_stride < 1)
			throw std::invalid_argument("plane fusion sample stride must be >= 1");
		const auto segmentation_start = std::chrono::steady_clock::now();
		std::vector<PlanePatchMap> patch_maps;
		patch_maps.reserve(num_images);
		for (int i = 0; i < num_images; ++i) {
			std::cout << "Segmenting planes in image " << std::setw(8) << std::setfill('0') << i << "...\n";
			patch_maps.emplace_back(BuildPlanePatchMap(depths[i], normals[i], cameras[i]));
		}
		const auto segmentation_end = std::chrono::steady_clock::now();
		std::cout << "Plane segmentation time: " << std::chrono::duration<double>(segmentation_end - segmentation_start).count() << " s\n";

		const auto fusion_start = std::chrono::steady_clock::now();
		std::vector<PointList> plane_point_cloud;
		uint64_t patch_pixels_skipped = 0;
		uint64_t fallback_pixels_tested = 0;
		uint64_t fused_plane_patches = 0;
		std::vector<std::vector<uchar>> consumed_patches(num_images);
		for (int i = 0; i < num_images; ++i)
			consumed_patches[i].assign(patch_maps[i].patches.size(), 0);

		for (int i = 0; i < num_images; ++i) {
			std::cout << "Plane-fusing image " << std::setw(8) << std::setfill('0') << i << "...\n";
			const auto &problem = problems[i];
			const int ref_index = imageIdToindexMap[problem.ref_image_id];
			const int cols = depths[ref_index].cols;
			const int rows = depths[ref_index].rows;
			const int num_ngb = problem.src_image_ids.size();
			const PlanePatchMap &ref_map = patch_maps[ref_index];

			// Validate a plane range against a few representative locations once;
			// then emit the requested sample grid without repeating view checks per point.
			for (int patch_id = 0; patch_id < static_cast<int>(ref_map.patches.size()); ++patch_id) {
				if (consumed_patches[ref_index][patch_id]) continue;
				const PlanePatch &ref_patch = ref_map.patches[patch_id];
				std::vector<int> supporting_images;
				float support_score_sum = 0.0f;
				for (int j = 0; j < num_ngb; ++j) {
					const int src_index = imageIdToindexMap[problem.src_image_ids[j]];
					int best_patch_id = -1;
					int best_hits = 0;
					float best_score = 0.0f;
					for (int rep = 0; rep < 5; ++rep) {
						const int ref_pixel = ref_patch.sample_pixels[rep];
						bool duplicate = false;
						for (int earlier = 0; earlier < rep; ++earlier)
							if (ref_patch.sample_pixels[earlier] == ref_pixel) duplicate = true;
						if (duplicate) continue;
						const int ref_r = ref_pixel / cols;
						const int ref_c = ref_pixel % cols;
					if (use_block && blocks[ref_index].at<uchar>(ref_r, ref_c) < 128) continue;
					float sample_depth = 0.0f;
					cv::Vec3f world_point;
					if (!IntersectPlaneAtPixel(ref_patch, cameras[ref_index], ref_c, ref_r,
							sample_depth, &world_point)) continue;
					const float3 sample = make_float3(world_point[0], world_point[1], world_point[2]);
					float2 projected;
					float projected_depth;
					ProjectCamera(sample, cameras[src_index], projected, projected_depth);
					const int src_c = static_cast<int>(projected.x + 0.5f);
					const int src_r = static_cast<int>(projected.y + 0.5f);
					if (src_c < 0 || src_r < 0 || src_c >= depths[src_index].cols || src_r >= depths[src_index].rows ||
						masks[src_index].at<uchar>(src_r, src_c) == 1) continue;
					const int src_patch_id = patch_maps[src_index].labels.at<int>(src_r, src_c);
					if (src_patch_id < 0 || consumed_patches[src_index][src_patch_id]) continue;
					const PlanePatch &src_patch = patch_maps[src_index].patches[src_patch_id];
					const float normal_cosine = std::fabs(ref_patch.normal.dot(src_patch.normal));
					if (!IsFiniteFloat(normal_cosine) || normal_cosine < std::cos(10.0f * static_cast<float>(M_PI) / 180.0f)) continue;
					const float plane_distance = std::fabs(src_patch.normal.dot(cv::Vec3f(sample.x, sample.y, sample.z)) + src_patch.offset);
					const float distance_tolerance = std::max(0.005f, 0.002f * projected_depth);
					if (!IsFiniteFloat(plane_distance) || plane_distance > distance_tolerance) continue;
					const float score = std::exp(-10.0f * std::acos(std::min(1.0f, normal_cosine)) -
						0.2f * plane_distance / distance_tolerance);
					if (src_patch_id == best_patch_id) {
						++best_hits;
						best_score += score;
					} else if (best_patch_id < 0) {
						best_patch_id = src_patch_id;
						best_hits = 1;
						best_score = score;
					}
				}
				if (best_patch_id >= 0 && best_hits >= 2) {
					supporting_images.push_back(src_index);
					support_score_sum += best_score / best_hits;
				}
			}
			const int seed_pixel = ref_patch.sample_pixels[0];
			const int seed_r = seed_pixel / cols, seed_c = seed_pixel % cols;
			const float support_factor = weaks[ref_index].at<uchar>(seed_r, seed_c) == WEAK ? 0.45f : 0.3f;
			if (supporting_images.empty() || support_score_sum <= support_factor * supporting_images.size()) continue;

			consumed_patches[ref_index][patch_id] = 1;
			++fused_plane_patches;
			bool emitted_patch_sample = false;
			for (int r = ref_patch.min_y; r <= ref_patch.max_y; ++r) {
				for (int c = ref_patch.min_x; c <= ref_patch.max_x; ++c) {
					if (ref_map.labels.at<int>(r, c) != patch_id ||
						(r % plane_sample_stride != 0 || c % plane_sample_stride != 0)) continue;
					if (masks[ref_index].at<uchar>(r, c) == 1) continue;
					if (use_block && blocks[ref_index].at<uchar>(r, c) < 128) continue;
					const float sample_depth = depths[ref_index].at<float>(r, c);
					if (!IsFiniteFloat(sample_depth) || sample_depth <= 0.0f) continue;
					const cv::Vec3f sample_normal = normals[ref_index].at<cv::Vec3f>(r, c);
					if (!IsValidNormal(sample_normal)) continue;
					const float3 sample_world = Get3DPointonWorld(c, r, sample_depth, cameras[ref_index]);
					PointList point3D;
					point3D.coord = sample_world;
					point3D.normal = make_float3(sample_normal[0], sample_normal[1], sample_normal[2]);
					const cv::Vec3b color = images[ref_index].at<cv::Vec3b>(r, c);
					point3D.color = make_float3(color[0], color[1], color[2]);
					plane_point_cloud.emplace_back(point3D);
					for (int src_index : supporting_images) {
						float2 projected;
						float projected_depth;
						ProjectCamera(sample_world, cameras[src_index], projected, projected_depth);
						const int src_c = static_cast<int>(projected.x + 0.5f);
						const int src_r = static_cast<int>(projected.y + 0.5f);
						if (src_c >= 0 && src_r >= 0 && src_c < masks[src_index].cols && src_r < masks[src_index].rows)
							masks[src_index].at<uchar>(src_r, src_c) = 1;
					}
					emitted_patch_sample = true;
				}
			}
			if (!emitted_patch_sample) {
				const int pixel = ref_patch.sample_pixels[0];
				const int r = pixel / cols, c = pixel % cols;
				const float sample_depth = depths[ref_index].at<float>(r, c);
				const cv::Vec3f sample_normal = normals[ref_index].at<cv::Vec3f>(r, c);
				const bool inside_block = !use_block || blocks[ref_index].at<uchar>(r, c) >= 128;
				if (inside_block && masks[ref_index].at<uchar>(r, c) == 0 &&
					IsFiniteFloat(sample_depth) && sample_depth > 0.0f && IsValidNormal(sample_normal)) {
					const float3 sample_world = Get3DPointonWorld(c, r, sample_depth, cameras[ref_index]);
					PointList point3D;
					point3D.coord = sample_world;
					point3D.normal = make_float3(sample_normal[0], sample_normal[1], sample_normal[2]);
					const cv::Vec3b color = images[ref_index].at<cv::Vec3b>(r, c);
					point3D.color = make_float3(color[0], color[1], color[2]);
					plane_point_cloud.emplace_back(point3D);
					for (int src_index : supporting_images) {
						float2 projected;
						float projected_depth;
						ProjectCamera(sample_world, cameras[src_index], projected, projected_depth);
						const int src_c = static_cast<int>(projected.x + 0.5f);
						const int src_r = static_cast<int>(projected.y + 0.5f);
						if (src_c >= 0 && src_r >= 0 && src_c < masks[src_index].cols && src_r < masks[src_index].rows)
							masks[src_index].at<uchar>(src_r, src_c) = 1;
					}
				}
			}
		}

			for (int r = 0; r < rows; ++r) {
				for (int c = 0; c < cols; ++c) {
					if (use_block && blocks[ref_index].at<uchar>(r, c) < 128) continue;
					if (masks[ref_index].at<uchar>(r, c) == 1) continue;
					const int patch_id = ref_map.labels.at<int>(r, c);
					if (patch_id >= 0 && consumed_patches[ref_index][patch_id]) {
						++patch_pixels_skipped;
						continue;
					}
					float ref_depth = depths[ref_index].at<float>(r, c);
					if (!IsFiniteFloat(ref_depth) || ref_depth <= 0.0f) continue;
					cv::Vec3f ref_normal = normals[ref_index].at<cv::Vec3f>(r, c);
					if (!IsValidNormal(ref_normal)) continue;

					const float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[ref_index]);
					++fallback_pixels_tested;
					int num_consistent = 0;
					float dynamic_consistency = 0.0f;
					std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
					for (int j = 0; j < num_ngb; ++j) {
						const int src_index = imageIdToindexMap[problem.src_image_ids[j]];
						float2 point;
						float proj_depth;
						ProjectCamera(PointX, cameras[src_index], point, proj_depth);
						const int src_r = int(point.y + 0.5f);
						const int src_c = int(point.x + 0.5f);
						if (src_c < 0 || src_c >= depths[src_index].cols || src_r < 0 || src_r >= depths[src_index].rows ||
							masks[src_index].at<uchar>(src_r, src_c) == 1) continue;

						const float src_depth = depths[src_index].at<float>(src_r, src_c);
						if (!IsFiniteFloat(src_depth) || src_depth <= 0.0f) continue;
						const cv::Vec3f src_normal = normals[src_index].at<cv::Vec3f>(src_r, src_c);
						if (!IsValidNormal(src_normal)) continue;
						const float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_index]);
					float2 tmp_pt;
					float reproj_depth;
					ProjectCamera(tmp_X, cameras[ref_index], tmp_pt, reproj_depth);
					const float reproj_error = std::sqrt((c - tmp_pt.x) * (c - tmp_pt.x) + (r - tmp_pt.y) * (r - tmp_pt.y));
					const float relative_depth_diff = std::fabs(proj_depth - src_depth) / std::max(1e-6f, proj_depth);
					const float angle = GetAngle(ref_normal, src_normal);
					if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
						used_list[j] = make_int2(src_c, src_r);
						const float score = reproj_error + 200.0f * relative_depth_diff + angle * 10.0f;
						dynamic_consistency += std::exp(-score);
						++num_consistent;
					}
				}
				const float factor = (weaks[ref_index].at<uchar>(r, c) == WEAK ? 0.45f : 0.3f);
				if (num_consistent < 1 || dynamic_consistency <= factor * num_consistent) continue;

				PointList point3D;
				point3D.coord = PointX;
				point3D.normal = make_float3(ref_normal[0], ref_normal[1], ref_normal[2]);
				float consistent_color[3] = {
					static_cast<float>(images[ref_index].at<cv::Vec3b>(r, c)[0]),
					static_cast<float>(images[ref_index].at<cv::Vec3b>(r, c)[1]),
					static_cast<float>(images[ref_index].at<cv::Vec3b>(r, c)[2])};
				for (int j = 0; j < num_ngb; ++j) {
					if (used_list[j].x < 0) continue;
					const int src_index = imageIdToindexMap[problem.src_image_ids[j]];
					masks[src_index].at<uchar>(used_list[j].y, used_list[j].x) = 1;
					const cv::Vec3b color = images[src_index].at<cv::Vec3b>(used_list[j].y, used_list[j].x);
					for (int channel = 0; channel < 3; ++channel) consistent_color[channel] += color[channel];
				}
			for (float &channel : consistent_color) channel /= (num_consistent + 1);
			point3D.color = make_float3(consistent_color[0], consistent_color[1], consistent_color[2]);
			plane_point_cloud.emplace_back(point3D);
				}
			}
		}
		const auto fusion_end = std::chrono::steady_clock::now();
		std::cout << "Plane fusion time: " << std::chrono::duration<double>(fusion_end - fusion_start).count()
			<< " s; fused " << fused_plane_patches << " plane patches and emitted "
			<< plane_point_cloud.size() << " points; tested " << fallback_pixels_tested
			<< " fallback pixels and skipped " << patch_pixels_skipped
			<< " pixels represented by accepted plane ranges.\n";
		const path ply_path = dense_folder / path("DPE") / path("DPE.ply");
		ExportPointCloud(ply_path, plane_point_cloud);
		const path fused_path = dense_folder / path("DPE") / path("fused.ply");
		if (!ExportFusedPointCloud(fused_path, plane_point_cloud))
			throw std::runtime_error("Failed to write " + fused_path.string());
		return;
	}
	std::vector<PointList> PointCloud;
	PointCloud.clear();
	uint64_t simple_candidates = 0;
	uint64_t simple_points_skipped = 0;

	for (int i = 0; i < num_images; ++i) {
		std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const auto &problem = problems[i];
		int ref_index = imageIdToindexMap[problem.ref_image_id];
		const int cols = depths[ref_index].cols;
		const int rows = depths[ref_index].rows;
		int num_ngb = problem.src_image_ids.size();
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (use_block && blocks[ref_index].at<uchar>(r, c) < 128) {
					continue;
				}

				if (masks[ref_index].at<uchar>(r, c) == 1) {
					continue;
				}

				float ref_depth = depths[ref_index].at<float>(r, c);
				if (!IsFiniteFloat(ref_depth) || ref_depth <= 0.0f)
					continue;
				if (simple_region_stride > 1 &&
					((r % simple_region_stride) != 0 || (c % simple_region_stride) != 0)) {
					++simple_candidates;
					if (IsStablePlanarNeighbourhood(depths[ref_index], normals[ref_index], cv::Mat(), cameras[ref_index], r, c)) {
						++simple_points_skipped;
						continue;
					}
				}
				const cv::Vec3f ref_normal = normals[ref_index].at<cv::Vec3f>(r, c);
				if (!IsValidNormal(ref_normal)) continue;
				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[ref_index]);
				float3 consistent_Point = PointX;
				int num_consistent = 0;
				float dynamic_consistency = 0.0f;
				std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
				for (int j = 0; j < num_ngb; ++j) {
					int src_index = imageIdToindexMap[problem.src_image_ids[j]];
					const int src_cols = depths[src_index].cols;
					const int src_rows = depths[src_index].rows;
					float2 point;
					float proj_depth;
					ProjectCamera(PointX, cameras[src_index], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_index].at<uchar>(src_r, src_c) == 1)
							continue;
						float src_depth = depths[src_index].at<float>(src_r, src_c);
						if (!IsFiniteFloat(src_depth) || src_depth <= 0.0f)
							continue;
						const cv::Vec3f src_normal = normals[src_index].at<cv::Vec3f>(src_r, src_c);
						if (!IsValidNormal(src_normal)) continue;
						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_index]);
						float2 tmp_pt;
						ProjectCamera(tmp_X, cameras[ref_index], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);

						if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
							used_list[j].x = src_c;
							used_list[j].y = src_r;
							float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
							dynamic_consistency += exp(-tmp_index);
							num_consistent++;
						}
					}
				}
				float factor = (weaks[ref_index].at<uchar>(r, c) == WEAK ? 0.45f : 0.3f);
				if (num_consistent >= 1 && (dynamic_consistency > factor * num_consistent)) {
					PointList point3D;
					point3D.coord = consistent_Point;
					point3D.normal = make_float3(ref_normal[0], ref_normal[1], ref_normal[2]);
					float consistent_Color[3] = { (float)images[ref_index].at<cv::Vec3b>(r, c)[0], (float)images[ref_index].at<cv::Vec3b>(r, c)[1], (float)images[ref_index].at<cv::Vec3b>(r, c)[2] };
					for (int j = 0; j < num_ngb; ++j) {
						if (used_list[j].x == -1)
							continue;
						int src_index = imageIdToindexMap[problem.src_image_ids[j]];
						masks[src_index].at<uchar>(used_list[j].y, used_list[j].x) = 1;
						const auto &color = images[src_index].at<cv::Vec3b>(used_list[j].y, used_list[j].x);
						consistent_Color[0] += color[0];
						consistent_Color[1] += color[1];
						consistent_Color[2] += color[2];
					}
					consistent_Color[0] /= (num_consistent + 1);
					consistent_Color[1] /= (num_consistent + 1);
					consistent_Color[2] /= (num_consistent + 1);
					point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
					PointCloud.emplace_back(point3D);
				}
			}
		}
	}
	path ply_path = dense_folder / path("DPE") / path("DPE.ply");
	ExportPointCloud(ply_path, PointCloud);
	const path fused_path = dense_folder / path("DPE") / path("fused.ply");
	if (!ExportFusedPointCloud(fused_path, PointCloud)) {
		throw std::runtime_error("Failed to write " + fused_path.string());
	}
	std::cout << "Saved " << PointCloud.size() << " points with normals to "
		<< fused_path.string() << std::endl;
	if (simple_region_stride > 1) {
		std::cout << "Adaptive point sampling (stride " << simple_region_stride << "): skipped "
			<< simple_points_skipped << " planar pixels / " << simple_candidates << " off-grid pixels checked ("
			<< (100.0 * simple_points_skipped / std::max<uint64_t>(1, simple_candidates)) << "%)." << std::endl;
	}
}

void RunFusion_TAT_Intermediate(const path &dense_folder, const std::vector<Problem> &problems)
{
	int num_images = problems.size();
	path image_folder = dense_folder / path("images");
	path cam_folder = dense_folder / path("cams");
	const float dist_base = 0.25f;
	const float depth_base = 1.0f / 3500.0f;

	const float angle_base = 0.06981317007977318f; // 4 degree
	const float angle_grad = 0.05235987755982988f; // 3 degree

	std::vector<cv::Mat> images;
	std::vector<Camera> cameras;
	std::vector<cv::Mat> depths;
	std::vector<cv::Mat> normals;
	std::vector<cv::Mat> masks;
	std::vector<cv::Mat> blocks;
	images.clear();
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	blocks.clear();
	std::unordered_map<int, int> imageIdToindexMap;

	path block_folder = dense_folder / path("blocks");
	bool use_block = false;
	if (exists(block_folder)) {
		use_block = true;
	}

	for (int i = 0; i < num_images; ++i) {
		const auto &problem = problems[i];
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		imageIdToindexMap.emplace(problem.ref_image_id, i);
		cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
		path cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		Camera camera;
		ReadCamera(cam_path, camera);

		path depth_path = problem.result_folder / path("depths.dmb");
		path normal_path = problem.result_folder / path("normals.dmb");
		cv::Mat depth, normal;
		ReadBinMat(depth_path, depth);
		ReadBinMat(normal_path, normal);

		if (use_block) {
			path block_path = block_folder / path("mask_" + std::to_string(problem.ref_image_id) + ".jpg");
			cv::Mat block_jpg = cv::imread(block_path.string(), cv::IMREAD_GRAYSCALE);
			if (block_jpg.empty()) throw std::runtime_error("Cannot read mask: " + block_path.string());
			cv::resize(block_jpg, block_jpg, depth.size(), 0, 0, cv::INTER_NEAREST);
			blocks.emplace_back(block_jpg);
		}

		cv::Mat scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		images.emplace_back(scaled_image);
		cameras.emplace_back(camera);
		depths.emplace_back(depth);
		normals.emplace_back(normal);
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
		masks.emplace_back(mask);
	}

	std::vector<PointList> PointCloud;
	PointCloud.clear();

	struct CostData
	{
		float dist;
		float depth;
		float angle;
		int src_r;
		int src_c;
		bool use;

		CostData () {
			dist = FLT_MAX;
			depth = FLT_MAX;
			angle = FLT_MAX;
		}
	};
	

	for (int i = 0; i < num_images; ++i) {
		std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const auto &problem = problems[i];
		int ref_index = imageIdToindexMap[problem.ref_image_id];
		const int cols = depths[ref_index].cols;
		const int rows = depths[ref_index].rows;
		int num_ngb = problem.src_image_ids.size();
		std::vector<CostData> diff(num_ngb, CostData());
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (use_block && blocks[ref_index].at<uchar>(r, c) < 128) {
					continue;
				}

				float ref_depth = depths[ref_index].at<float>(r, c);
				if (ref_depth <= 0.0)
					continue;
				const cv::Vec3f ref_normal = normals[ref_index].at<cv::Vec3f>(r, c);
				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[ref_index]);
				float3 consistent_Point = PointX;
				for (int j = 0; j < num_ngb; ++j) {
					int src_index = imageIdToindexMap[problem.src_image_ids[j]];
					const int src_cols = depths[src_index].cols;
					const int src_rows = depths[src_index].rows;
					float2 point;
					float proj_depth;
					ProjectCamera(PointX, cameras[src_index], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_index].at<uchar>(src_r, src_c) == 1)
							continue;
						float src_depth = depths[src_index].at<float>(src_r, src_c);
						if (src_depth <= 0.0)
							continue;
						const cv::Vec3f src_normal = normals[src_index].at<cv::Vec3f>(src_r, src_c);
						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_index]);
						float2 tmp_pt;
						ProjectCamera(tmp_X, cameras[ref_index], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);
						diff[j].dist = reproj_error;
						diff[j].depth = relative_depth_diff;
						diff[j].angle = angle;
						diff[j].src_r = src_r;
						diff[j].src_c = src_c;
					}
				}
				for (int k = 2; k <= num_ngb; ++k) {
					int count = 0;
					for (int j = 0; j < num_ngb; ++j) {
						diff[j].use = false;
						if (diff[j].dist < k * dist_base && diff[j].depth < k * depth_base && diff[j].angle < (k * angle_grad + angle_base)) {
							count++;
							diff[j].use = true;
						}
					}
					if (count >= k) {
						PointList point3D;						
						float consistent_Color[3] = { (float)images[ref_index].at<cv::Vec3b>(r, c)[0], (float)images[ref_index].at<cv::Vec3b>(r, c)[1], (float)images[ref_index].at<cv::Vec3b>(r, c)[2] };
						for (int j = 0; j < num_ngb; ++j) {
							if (diff[j].use) {
								int src_index = imageIdToindexMap[problem.src_image_ids[j]];
								consistent_Color[0] += (float)images[src_index].at<cv::Vec3b>(diff[j].src_r, diff[j].src_c)[0];
								consistent_Color[1] += (float)images[src_index].at<cv::Vec3b>(diff[j].src_r, diff[j].src_c)[1];
								consistent_Color[2] += (float)images[src_index].at<cv::Vec3b>(diff[j].src_r, diff[j].src_c)[2];
							}
						}
						consistent_Color[0] /= (count + 1.0f);
						consistent_Color[1] /= (count + 1.0f);
						consistent_Color[2] /= (count + 1.0f);

						point3D.coord = consistent_Point;
						point3D.normal = make_float3(ref_normal[0], ref_normal[1], ref_normal[2]);
						point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
						PointCloud.emplace_back(point3D);
						masks[ref_index].at<uchar>(r, c) = 1;
						break;
					}
				}
			}
		}
	}
	path ply_path = dense_folder / path("DPE") / path("DPE.ply");
	ExportPointCloud(ply_path, PointCloud);
	const path fused_path = dense_folder / path("DPE") / path("fused.ply");
	if (!ExportFusedPointCloud(fused_path, PointCloud)) {
		throw std::runtime_error("Failed to write " + fused_path.string());
	}
	std::cout << "Saved " << PointCloud.size() << " points with normals to "
		<< fused_path.string() << std::endl;
}

void RunFusion_TAT_advanced(const path &dense_folder, const std::vector<Problem> &problems)
{
	int num_images = problems.size();
	path image_folder = dense_folder / path("images");
	path cam_folder = dense_folder / path("cams");
	const float dist_base = 0.25f;
	const float depth_base = 1.0f / 3000.0f;

	std::vector<cv::Mat> images;
	std::vector<Camera> cameras;
	std::vector<cv::Mat> depths;
	std::vector<cv::Mat> normals;
	std::vector<cv::Mat> masks;
	std::vector<cv::Mat> blocks;
	images.clear();
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	blocks.clear();
	std::unordered_map<int, int> imageIdToindexMap;

	path block_folder = dense_folder / path("blocks");
	bool use_block = false;
	if (exists(block_folder)) {
		use_block = true;
	}

	for (int i = 0; i < num_images; ++i) {
		const auto &problem = problems[i];
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		imageIdToindexMap.emplace(problem.ref_image_id, i);
		cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
		path cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		Camera camera;
		ReadCamera(cam_path, camera);

		path depth_path = problem.result_folder / path("depths.dmb");
		path normal_path = problem.result_folder / path("normals.dmb");
		cv::Mat depth, normal;
		ReadBinMat(depth_path, depth);
		ReadBinMat(normal_path, normal);

		if (use_block) {
			path block_path = block_folder / path("mask_" + std::to_string(problem.ref_image_id) + ".jpg");
			cv::Mat block_jpg = cv::imread(block_path.string(), cv::IMREAD_GRAYSCALE);
			if (block_jpg.empty()) throw std::runtime_error("Cannot read mask: " + block_path.string());
			cv::resize(block_jpg, block_jpg, depth.size(), 0, 0, cv::INTER_NEAREST);
			blocks.emplace_back(block_jpg);
		}

		cv::Mat scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		images.emplace_back(scaled_image);
		cameras.emplace_back(camera);
		depths.emplace_back(depth);
		normals.emplace_back(normal);
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
		masks.emplace_back(mask);
	}

	std::vector<PointList> PointCloud;
	PointCloud.clear();

	struct CostData
	{
		float dist;
		float depth;
		float angle;

		CostData () {
			dist = FLT_MAX;
			depth = FLT_MAX;
			angle = FLT_MAX;
		}
	};
	

	for (int i = 0; i < num_images; ++i) {
		std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const auto &problem = problems[i];
		int ref_index = imageIdToindexMap[problem.ref_image_id];
		const int cols = depths[ref_index].cols;
		const int rows = depths[ref_index].rows;
		int num_ngb = problem.src_image_ids.size();
		std::vector<CostData> diff(num_ngb, CostData());
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (use_block && blocks[ref_index].at<uchar>(r, c) < 128) {
					continue;
				}

				float ref_depth = depths[ref_index].at<float>(r, c);
				if (ref_depth <= 0.0)
					continue;
				const cv::Vec3f ref_normal = normals[ref_index].at<cv::Vec3f>(r, c);
				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[ref_index]);
				float3 consistent_Point = PointX;
				float consistent_Color[3] = { (float)images[ref_index].at<cv::Vec3b>(r, c)[0], (float)images[ref_index].at<cv::Vec3b>(r, c)[1], (float)images[ref_index].at<cv::Vec3b>(r, c)[2] };

				for (int j = 0; j < num_ngb; ++j) {
					int src_index = imageIdToindexMap[problem.src_image_ids[j]];
					const int src_cols = depths[src_index].cols;
					const int src_rows = depths[src_index].rows;
					float2 point;
					float proj_depth;
					ProjectCamera(PointX, cameras[src_index], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_index].at<uchar>(src_r, src_c) == 1)
							continue;
						float src_depth = depths[src_index].at<float>(src_r, src_c);
						if (src_depth <= 0.0)
							continue;
						const cv::Vec3f src_normal = normals[src_index].at<cv::Vec3f>(src_r, src_c);
						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_index]);
						float2 tmp_pt;
						ProjectCamera(tmp_X, cameras[ref_index], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);
						diff[j].dist = reproj_error;
						diff[j].depth = relative_depth_diff;
						diff[j].angle = angle;
					}
				}
				for (int k = 2; k <= num_ngb; ++k) {
					int count = 0;
					for (int j = 0; j < num_ngb; ++j) {
						if (diff[j].dist < k * dist_base && diff[j].depth < k * depth_base) {
							count++;
						}
					}
					if (count >= k) {
						PointList point3D;
						point3D.coord = consistent_Point;
						point3D.normal = make_float3(ref_normal[0], ref_normal[1], ref_normal[2]);
						point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
						PointCloud.emplace_back(point3D);
						masks[ref_index].at<uchar>(r, c) = 1;
						break;
					}
				}
			}
		}
	}
	path ply_path = dense_folder / path("DPE") / path("DPE.ply");
	ExportPointCloud(ply_path, PointCloud);
	const path fused_path = dense_folder / path("DPE") / path("fused.ply");
	if (!ExportFusedPointCloud(fused_path, PointCloud)) {
		throw std::runtime_error("Failed to write " + fused_path.string());
	}
	std::cout << "Saved " << PointCloud.size() << " points with normals to "
		<< fused_path.string() << std::endl;
}

void ExportDepthImagePointCloud(
	const path& point_cloud_path, 
	const path& image_path,
	const path& cam_path,
	cv::Mat& depth,
	float depth_min,
	float depth_max
	) {
	std::vector<PointList> PointCloud;
	PointCloud.clear();

	cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);

	Camera camera;
	ReadCamera(cam_path, camera);

	cv::Mat scaled_image;
	RescaleImageAndCamera(image, scaled_image, depth, camera);

	for (int i = 0; i < depth.cols; i++) {
		for (int j = 0; j < depth.rows; j++) {
			if (depth.at<float>(j, i) < depth_min || depth.at<float>(j, i) > depth_max || isnan(depth.at<float>(j, i))) {
				continue;
			}

			PointList point3D;
			point3D.coord = Get3DPointonWorld(i, j, depth.at<float>(j, i), camera);
			point3D.color = { (float)scaled_image.at<cv::Vec3b>(j, i)[0], (float)scaled_image.at<cv::Vec3b>(j, i)[1], (float)scaled_image.at<cv::Vec3b>(j, i)[2] };
			PointCloud.push_back(point3D);
		}
	}

	ExportPointCloud(point_cloud_path, PointCloud);
}
