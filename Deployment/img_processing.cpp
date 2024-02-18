#include "img_processing.h"

/****
 * Function: ImgTransform
 * Description: Transforms an input image using various operations, including resizing, data type conversion,
 *              and mean/std-deviation normalization.
 *
 * @param input_img (const cv::Mat&): The input image to be transformed.
 * @param input_dim (std::vector<int64_t>): A vector specifying the desired dimensions of the transformed image.
 *                                          Format: [width, height, depth, channels]
 *
 * @return (cv::Mat): The transformed image after resizing, conversion, and normalization.
 ****/
cv::Mat	ImgTransform(const cv::Mat& input_img, std::vector<int64_t> input_dim)
{
	cv::Mat img = input_img.clone();

	int len = std::max(img.cols, img.rows);
	int bottom = len - img.rows;
	int right = len - img.cols;
	cv::copyMakeBorder(img, img, 0, bottom, 0, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	cv::resize(img, img, cv::Size(input_dim[3], input_dim[2]));
	img.convertTo(img, CV_32FC3);

	// Subtract mean and divide by standard deviation for each channel
	cv::Scalar mean, std_dev;
	cv::meanStdDev(img, mean, std_dev);
	cv::subtract(img, mean, img);
	cv::divide(img, std_dev, img);

	return img;
}

/**
 * Function: FlattenImageChannels
 * Description: Flattens the color channels of an input image into a single vector.
 *
 * This function takes an image and flattens its color channels into a one-dimensional vector.
 * It preserves the channel order and rows and columns of the image, resulting in a vector
 * where each element corresponds to a pixel value from the image.
 *
 * @param image (const cv::Mat&): The input image to be flattened.
 * @param input_dim (const std::vector<int64_t>&): A vector specifying the desired dimensions
 *                                                of the input image [channels, rows, cols].
 *
 * @return (std::vector<float>): A one-dimensional vector containing the flattened image data.
 *                             The vector size is equal to channels * rows * cols.
 */
std::vector<float> FlattenImageChannels(const cv::Mat& image, const std::vector<int64_t>& input_dim)
{
	std::vector<float> input_en;
	input_en.resize(static_cast<size_t>(input_dim[1] * input_dim[2] * input_dim[3]));

	for (auto channel = 0; channel < input_dim[1]; ++channel)
	{
		for (auto row = 0; row < input_dim[2]; ++row)
		{
			for (auto col = 0; col < input_dim[3]; ++col)
			{
				// Calculate the index for the flattened input_en vector
				size_t index = static_cast<size_t>(channel * input_dim[2] * input_dim[3] + row * input_dim[3] + col);

				// Access the pixel value from the image
				float pixel_value = image.at<cv::Vec3f>(row, col)[channel];

				// Assign the pixel value to the corresponding position in the input_en vector
				input_en[index] = pixel_value;
			}
		}
	}

	return input_en;
}