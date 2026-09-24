#include "main.h"
#include "DPE.h"

using namespace boost::filesystem;

void GenerateSampleList(const path &dense_folder, std::vector<Problem> &problems)
{
	path cluster_list_path = dense_folder / path("pair.txt");
	problems.clear();
	ifstream file(cluster_list_path);
	std::stringstream iss;
	std::string line;

	int num_images;
	iss.clear();
	std::getline(file, line);
	iss.str(line);
	iss >> num_images;

	for (int i = 0; i < num_images; ++i) {
		Problem problem;
		problem.index = i;
		problem.src_image_ids.clear();
		iss.clear();
		std::getline(file, line);
		iss.str(line);
		iss >> problem.ref_image_id;

		problem.dense_folder = dense_folder;
		problem.result_folder = dense_folder / path(OUT_NAME) / path(ToFormatIndex(problem.ref_image_id));
		create_directory(problem.result_folder);

		int num_src_images;
		iss.clear();
		std::getline(file, line);
		iss.str(line);
		iss >> num_src_images;
		for (int j = 0; j < num_src_images; ++j) {
			int id;
			float score;
			iss >> id >> score;
			if (score <= 0.0f) {
				continue;
			}
			problem.src_image_ids.push_back(id);
		}
		problems.push_back(problem);
	}
}

bool CheckImages(const std::vector<Problem> &problems) {
	if (problems.size() == 0) {
		return false;
	}
	path image_path = problems[0].dense_folder / path("images") / path(ToFormatIndex(problems[0].ref_image_id) + ".jpg");
	cv::Mat image = cv::imread(image_path.string());
	if (image.empty()) {
		return false;
	}
	const int width = image.cols;
	const int height = image.rows;
	for (size_t i = 1; i < problems.size(); ++i) {
		image_path = problems[i].dense_folder / path("images") / path(ToFormatIndex(problems[i].ref_image_id) + ".jpg");
		image = cv::imread(image_path.string());
		if (image.cols != width || image.rows != height) {
			return false;
		}
	}
	return true;
}

void GetProblemEdges(const Problem &problem) {
	std::cout << "Getting image edges: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
	int scale = 0;
	while((1 << scale) < problem.scale_size) scale++;

	path image_folder = problem.dense_folder / path("images");
	path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
	cv::Mat image_uint = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
	cv::Mat src_img;
	const cv::Size capped = LimitedImageSize(image_uint.cols, image_uint.rows, problem.params.max_image_size);
    if (capped != image_uint.size()) cv::resize(image_uint, image_uint, capped, 0, 0, cv::INTER_LINEAR);
    image_uint.convertTo(src_img, CV_32FC1);
	const float factor = 1.0f / (float)(problem.scale_size);
	const int new_cols = std::round(src_img.cols * factor);
	const int new_rows = std::round(src_img.rows * factor);
	cv::Mat scaled_image_float;
	cv::resize(src_img, scaled_image_float, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
	scaled_image_float.convertTo(src_img, CV_8UC1);
	std::cout << "size: " << new_cols << "x" << new_rows << "\n";

	if (problem.params.use_edge) {
		// path edge_path = problem.result_folder / path("edges.dmb");
		path edge_path = problem.result_folder / path("edges_max" + std::to_string(problem.params.max_image_size) + "_" + std::to_string(scale) + ".dmb");
		std::ifstream edge_file(edge_path.string());
		bool edge_exists = edge_file.good();
		edge_file.close();
		if (!edge_exists) {
			std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
			cv::Mat edge = EdgeSegment(scale, src_img, 0, true, problem.params.high_res_img);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << "Fine edge cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
			WriteBinMat(edge_path, edge);
			if (problem.show_medium_result) {
				path ref_image_edge_path = problem.result_folder / path("rawedge_" + std::to_string(scale) + ".jpg");
				cv::imwrite(ref_image_edge_path.string(), edge);
			}
		}
	}

	if (problem.params.use_label) {
		// path label_path = problem.result_folder / path("labels.dmb");
		path label_path = problem.result_folder / path("labels_max" + std::to_string(problem.params.max_image_size) + "_" + std::to_string(scale) + ".dmb");
		std::ifstream label_file(label_path.string());
		bool label_exists = label_file.good();
		label_file.close();
		if (!label_exists) {
			std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
			cv::Mat label = EdgeSegment(scale, image_uint, 1, false, problem.params.high_res_img);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << "Coarse edge cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
			WriteBinMat(label_path, label);
			if (problem.show_medium_result) {
				path ref_image_con_path = problem.result_folder / path("connect_" + std::to_string(scale) + ".jpg");
				cv::imwrite(ref_image_con_path.string(), EdgeSegment(scale, image_uint, -1, false, problem.params.high_res_img));
			}
		}
	}

	std::cout << "Getting image edges: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
}

int ComputeRoundNum(const std::vector<Problem> &problems) {
	if (problems.size() == 0) {
		return 0;
	}
	path image_path = problems[0].dense_folder / path("images") / path(ToFormatIndex(problems[0].ref_image_id) + ".jpg");
	cv::Mat image = cv::imread(image_path.string());
	if (image.empty()) {
		return 0;
	}
	const cv::Size capped = LimitedImageSize(image.cols, image.rows, problems[0].params.max_image_size);
	int max_size = MAX(capped.width, capped.height);
	int round_num = 1;
	while (max_size > 800) {
		max_size /= 2;
		round_num++;
	}
	return round_num;
}


float ProcessProblem(const Problem &problem) {
	std::cout << "Processing image: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
    std::cout << "iteration: " << problem.iteration << std::endl;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	DPE DPE(problem);
	DPE.InuputInitialization();
	DPE.SupportInitialization();
	DPE.CudaSpaceInitialization();
	DPE.SetDataPassHelperInCuda();
	DPE.RunPatchMatch();

	int width = DPE.GetWidth(), height = DPE.GetHeight();
	cv::Mat depth = cv::Mat(height, width, CV_32FC1);
	cv::Mat normal = cv::Mat(height, width, CV_32FC3);
	cv::Mat pixel_states = DPE.GetPixelStates();
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			float4 plane_hypothesis = DPE.GetPlaneHypothesis(r, c);
			depth.at<float>(r, c) = plane_hypothesis.w;
			if (depth.at<float>(r, c) < DPE.GetDepthMin() || depth.at<float>(r, c) > DPE.GetDepthMax()) {
				depth.at<float>(r, c) = 0;
				pixel_states.at<uchar>(r, c) = UNKNOWN;
			}
			normal.at<cv::Vec3f>(r, c) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
		}
	}
	
	path depth_path = problem.result_folder / path("depths.dmb");
	WriteBinMat(depth_path, depth);
	path normal_path = problem.result_folder / path("normals.dmb");
	WriteBinMat(normal_path, normal);
	path weak_path = problem.result_folder / path("weak.bin");
	WriteBinMat(weak_path, pixel_states);
	path selected_view_path = problem.result_folder / path("selected_views.bin");
	WriteBinMat(selected_view_path, DPE.GetSelectedViews());

	if (problem.show_medium_result) {
		path depth_img_path = problem.result_folder / path("depth_" + std::to_string(problem.iteration) + ".jpg");
		path normal_img_path = problem.result_folder / path("normal_" + std::to_string(problem.iteration) + ".jpg");
		path weak_img_path = problem.result_folder / path("weak_" + std::to_string(problem.iteration) + ".jpg");
		ShowDepthMap(depth_img_path, depth, DPE.GetDepthMin(), DPE.GetDepthMax());
		ShowNormalMap(normal_img_path, normal);
		ShowWeakImage(weak_img_path, pixel_states);

		if ((problem.iteration + 1) % 4 == 0) {
			path image_folder = problem.dense_folder / path("images");
			path cam_folder = problem.dense_folder / path("cams");
			path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
			path cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
			path point_cloud_path = problem.result_folder / path("point_" + std::to_string(problem.iteration) + ".ply");
			// path point_cloud_path = problem.result_folder / path("point_test_" + std::to_string(problem.iteration) + ".ply");

			// for (int r = 0; r < height; ++r) for (int c = 0; c < width; ++c) if (pixel_states.at<uchar>(r, c) != STRONG) depth.at<float>(r, c) = 0;
			// ExportDepthImagePointCloud(point_cloud_path, image_path, cam_path, depth, DPE.GetDepthMin(), DPE.GetDepthMax());
			// remove(point_cloud_path);
		}
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Processing image: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
	std::cout << "Cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	return DPE.GetAdaptiveFrozenFraction();
}

bool ValidateFusionInputs(const std::vector<Problem> &problems) {
	bool valid = true;
	for (const auto &problem : problems) {
		for (const char *filename : {"depths.dmb", "normals.dmb", "weak.bin"}) {
			const path input_path = problem.result_folder / path(filename);
			if (!is_regular_file(input_path) || file_size(input_path) == 0) {
				std::cerr << "Missing or empty fusion input: "
				          << input_path.string() << std::endl;
				valid = false;
			}
		}
	}
	return valid;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "USAGE: DPE dense_folder [gpu_index] [--fuse] "
                     "[--geometric-anchor-cost] [--adaptive-refinement] "
					 "[--adaptive-refinement-aggressiveness 1|2|3] "
					 "[--adaptive-refinement-early-stop FRACTION] "
                     "[--adaptive-point-sampling] [--simple-region-stride N] "
                     "[--start-round N] "
                     "[--max-image-size N]\n";
        return EXIT_FAILURE;
    }
    path dense_folder(argv[1]);
    int gpu_index = 0, max_image_size = 3200;
	bool fuse_only = false;
    bool geometric_anchor_cost = false;
    bool adaptive_refinement = false;
    int adaptive_refinement_aggressiveness = 1;
    float adaptive_refinement_early_stop = 0.0f;
    bool adaptive_point_sampling = false;
    int simple_region_stride = 2;
    int start_round = 0;
    int arg = 2;
    if (arg < argc && std::string(argv[arg]).find("--") != 0) {
        gpu_index = std::atoi(argv[arg++]); // Preserve legacy positional argument.
    }
    try {
        while (arg < argc) {
            const std::string option(argv[arg++]);
            if (option == "--fuse") fuse_only = true;
            else if (option == "--geometric-anchor-cost") geometric_anchor_cost = true;
            else if (option == "--adaptive-refinement") adaptive_refinement = true;
            else if (option == "--adaptive-refinement-aggressiveness" && arg < argc) {
                const std::string value(argv[arg++]);
                size_t end = 0;
                adaptive_refinement_aggressiveness = std::stoi(value, &end);
                if (end != value.size() || adaptive_refinement_aggressiveness < 1 ||
                    adaptive_refinement_aggressiveness > 3)
                    throw std::invalid_argument("adaptive-refinement-aggressiveness must be 1, 2, or 3");
            }
            else if (option == "--adaptive-refinement-early-stop" && arg < argc) {
                const std::string value(argv[arg++]);
                size_t end = 0;
                adaptive_refinement_early_stop = std::stof(value, &end);
                if (end != value.size() || !(adaptive_refinement_early_stop >= 0.0f &&
                    adaptive_refinement_early_stop <= 1.0f))
                    throw std::invalid_argument("adaptive-refinement-early-stop must be between 0 and 1");
            }
            else if (option == "--adaptive-point-sampling") adaptive_point_sampling = true;
            else if (option == "--simple-region-stride" && arg < argc) {
                const std::string value(argv[arg++]);
                size_t end = 0;
                simple_region_stride = std::stoi(value, &end);
                if (end != value.size() || simple_region_stride < 1)
                    throw std::invalid_argument("simple-region-stride must be >= 1");
            }
            else if (option == "--start-round" && arg < argc) {
                const std::string value(argv[arg++]);
                size_t end = 0;
                start_round = std::stoi(value, &end);
                if (end != value.size() || start_round < 0)
                    throw std::invalid_argument("start-round must be >= 0");
            }
            else if (option == "--max-image-size" && arg < argc) {
                const std::string value(argv[arg++]);
                size_t end = 0;
                max_image_size = std::stoi(value, &end);
                if (end != value.size() || max_image_size < 0) throw std::invalid_argument("invalid size");
            } else throw std::invalid_argument("unknown or incomplete option: " + option);
        }
        if (adaptive_refinement_early_stop > 0.0f && !adaptive_refinement)
            throw std::invalid_argument("adaptive-refinement-early-stop requires --adaptive-refinement");
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    path output_folder = dense_folder / path(OUT_NAME);
    create_directory(output_folder);
    std::cout << "Maximum image size: " << max_image_size << " (0 = original)" << std::endl;
	// generate problems
	std::vector<Problem> problems;
	GenerateSampleList(dense_folder, problems);
	for (auto &problem : problems) {
        problem.params.geometric_anchor_cost = geometric_anchor_cost;
        problem.params.adaptive_refinement = adaptive_refinement;
        problem.params.adaptive_refinement_aggressiveness = adaptive_refinement_aggressiveness;
        problem.params.max_image_size = max_image_size;
    }
	std::cout << "Geometric anchor cost: " << geometric_anchor_cost << std::endl;
	std::cout << "Adaptive refinement: " << adaptive_refinement << " (aggressiveness "
		<< adaptive_refinement_aggressiveness << ", early-stop threshold "
		<< adaptive_refinement_early_stop << ")" << std::endl;
	std::cout << "Adaptive point sampling: " << adaptive_point_sampling
	          << " (simple-region stride " << simple_region_stride << ")" << std::endl;
	if (!CheckImages(problems)) {
		std::cerr << "Images may error, check it!\n";
		return EXIT_FAILURE;
	}
	int num_images = problems.size();
	std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;
	if (fuse_only) {
		std::cout << "Fuse-only mode: skipping edge detection and depth estimation." << std::endl;
		if (!ValidateFusionInputs(problems)) {
			std::cerr << "Fusion inputs are incomplete. Run depth estimation first." << std::endl;
			return EXIT_FAILURE;
		}
		RunFusion(dense_folder, problems, adaptive_point_sampling ? simple_region_stride : 1);
		std::cout << "Fusion done. Intermediate depth and normal files were preserved.\n";
		return EXIT_SUCCESS;
	}

	cudaSetDevice(gpu_index);

	int round_num = ComputeRoundNum(problems);
	if (start_round >= round_num) {
		std::cerr << "Start round " << start_round << " is outside the pyramid (0.."
			<< (round_num - 1) << ").\n";
		return EXIT_FAILURE;
	}
	for (auto &problem : problems) {
		problem.params.max_scale_size = 1;
		for (int i = 0; i < round_num; ++i) {
			problem.scale_size = static_cast<int>(std::pow(2, round_num - 1 - i)); // scale 
			GetProblemEdges(problem); // 注意要先得到 scale_size
			problem.params.max_scale_size = MAX(problem.scale_size, problem.params.max_scale_size);
		}
	}

	std::cout << "Round nums: " << round_num << std::endl;
	int iteration_index = start_round * 4;
	std::vector<bool> early_stopped(problems.size(), false);
	if (start_round > 0) {
		std::cout << "Resuming from pyramid round " << start_round
			<< " using the saved preceding-scale depth maps." << std::endl;
	}
	for (int i = start_round; i < round_num; ++i) {
		std::fill(early_stopped.begin(), early_stopped.end(), false);
		for (auto &problem : problems) {
			problem.iteration = iteration_index;
			problem.scale_size = static_cast<int>(std::pow(2, round_num - 1 - i)); // scale 
			problem.params.scale_size = problem.scale_size;
			{
				auto &params = problem.params;
				if (i == 0) {
					params.state = FIRST_INIT;
					params.use_APD = false;
					params.use_edge = false;
				} else {
					params.state = REFINE_INIT;
					params.use_APD = true;
					params.use_edge = true;
					params.ransac_threshold = 0.01 - i * 0.00125;
					params.rotate_time = MIN(static_cast<int>(std::pow(2, i)), 4);
				}
				params.geom_consistency = false;
				params.max_iterations = 3;
				params.weak_peak_radius = 6;
			}
			const float frozen_fraction = ProcessProblem(problem);
			if (adaptive_refinement_early_stop > 0.0f &&
				frozen_fraction >= adaptive_refinement_early_stop) {
				early_stopped[problem.index] = true;
				std::cout << "Adaptive early stop: image " << ToFormatIndex(problem.ref_image_id)
					<< " has " << (100.0f * frozen_fraction) << "% frozen at scale "
					<< problem.scale_size << "; skipping remaining iterations at this scale.\n";
			}
		}
		iteration_index++;
		for (int j = 0; j < 3; ++j) {
			for (auto &problem : problems) {
				if (early_stopped[problem.index]) {
					std::cout << "Adaptive early stop: skip image " << ToFormatIndex(problem.ref_image_id)
						<< " at iteration " << iteration_index << ".\n";
					continue;
				}
				problem.iteration = iteration_index;
				problem.scale_size = static_cast<int>(std::pow(2, round_num - 1 - i)); // scale 
				problem.params.scale_size = problem.scale_size;
				{
					auto &params = problem.params;
					params.state = REFINE_ITER;
					if (i == 0) {
						params.use_APD = false;
						params.use_edge = false;
					} else {
						params.use_APD = true;
						params.use_edge = true;
					}
					params.ransac_threshold = 0.01 - i * 0.00125;
					params.rotate_time = MIN(static_cast<int>(std::pow(2, i)), 4);
					params.geom_consistency = true;
					params.max_iterations = 3;
					params.weak_peak_radius = MAX(4 - 2 * j, 2);
				}
				const float frozen_fraction = ProcessProblem(problem);
				if (adaptive_refinement_early_stop > 0.0f &&
					frozen_fraction >= adaptive_refinement_early_stop) {
					early_stopped[problem.index] = true;
					std::cout << "Adaptive early stop: image " << ToFormatIndex(problem.ref_image_id)
						<< " has " << (100.0f * frozen_fraction) << "% frozen at scale "
						<< problem.scale_size << "; skipping remaining iterations at this scale.\n";
				}
			}
			iteration_index++;
		}
		std::cout << "Round: " << i << " done\n";
	}

	RunFusion(dense_folder, problems, adaptive_point_sampling ? simple_region_stride : 1);
	{// delete files
		for (size_t i = 0; i < problems.size(); ++i) {
			const auto &problem = problems[i];
			// These three files are the persistent input of --fuse. Keep them
			// regardless of geometric_anchor_cost or visualization settings.
			// They are deliberately not part of this cleanup block.
			// remove(problem.result_folder / path("weak.bin"));
			// remove(problem.result_folder / path("depths.dmb"));
			// remove(problem.result_folder / path("normals.dmb"));
			remove(problem.result_folder / path("selected_views.bin"));
			remove(problem.result_folder / path("neighbour.bin")); 
			remove(problem.result_folder / path("neighbour_map.bin"));
			for (int j = 0; j < round_num; j++) {
				remove(problem.result_folder / path("edges_max" + std::to_string(problem.params.max_image_size) + "_" + std::to_string(j) + ".dmb"));
				remove(problem.result_folder / path("labels_max" + std::to_string(problem.params.max_image_size) + "_" + std::to_string(j) + ".dmb"));
			}
		}
	}
	std::cout << "All done\n";
	return EXIT_SUCCESS;
}
