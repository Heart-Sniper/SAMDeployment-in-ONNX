#include <opencv2/opencv.hpp> 

cv::Mat	ImgTransform(const cv::Mat& input_img, std::vector<int64_t> input_dim);
std::vector<float> FlattenImageChannels(const cv::Mat& image, const std::vector<int64_t>& input_dim);