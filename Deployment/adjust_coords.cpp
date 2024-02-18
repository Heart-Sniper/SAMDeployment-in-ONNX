# include "adjust_coords.h"

/****
 * Function: CalculateResizeDim
 * Description: Calculates new dimensions for resizing an image while preserving its aspect ratio.
 *
 * @param old_h (int): The original image height.
 * @param old_w (int): The original image width.
 * @param longside_length (int): The length of the longer side of the target image.
 *
 * @return (std::tuple<int, int>): A tuple containing the new height and new width.
 ****/
std::tuple<int, int> CalculateResizeDim(int old_h, int old_w, int longside_length)
{
	float scale = static_cast<float>(longside_length) / std::max(old_h, old_w);
	//round the new dimensions
	int new_h = static_cast<int>(old_h * scale + 0.5);
	int new_w = static_cast<int>(old_w * scale + 0.5);
	return std::make_tuple(new_h, new_w);
}

/****
 * Function: AdjustCoords
 * Description: Adjusts a vector of coordinates to match a target image size.
 *
 * @param coords (const std::vector<std::tuple<float, float>>&): The original coordinates.
 * @param original_size (const std::tuple<int, int>&): The original image size (height, width).
 * @param target_length (int): The length of the longer side of the target image.
 *
 * @return (std::vector<std::tuple<float, float>>): A vector of adjusted coordinates.
 ****/
std::vector<std::tuple<float, float>> AdjustCoords(const std::vector<std::tuple<float, float>>& coords,
	const std::tuple<int, int>& original_size,
	int target_length)
{
	int old_h = std::get<0>(original_size);
	int old_w = std::get<1>(original_size);
	int new_h, new_w;
	std::tie(new_h, new_w) = CalculateResizeDim(old_h, old_w, target_length);

	std::vector<std::tuple<float, float>> transformed_coords = coords;
	for (auto& coord : transformed_coords)
	{
		std::get<0>(coord) = std::get<0>(coord) * (new_w / static_cast<float>(old_w));
		std::get<1>(coord) = std::get<1>(coord) * (new_h / static_cast<float>(old_h));
	}

	return transformed_coords;
}

/****
 * Function: AdjustBox
 * Description: Adjusts a vector of bounding boxes to match a target image size.
 *
 * @param boxes (const std::vector<std::vector<float>>&): The original bounding boxes.
 * @param original_size (const std::tuple<int, int>&): The original image size (height, width).
 * @param target_length (int): The length of the longer side of the target image.
 *
 * @return (std::vector<std::vector<float>>): A vector of adjusted bounding boxes.
 ****/

std::vector<std::vector<float>> AdjustBox(const std::vector<std::vector<float>>& boxes,
	const std::tuple<int, int>& original_size,
	int target_length)
{
	std::vector<std::vector<float>> transformed_boxes;
	for (const auto& box : boxes)
	{
		std::vector<std::tuple<float, float>> box_points = { {box[0], box[1]}, {box[2], box[3]} };
		std::vector<std::tuple<float, float>> transformed_box = AdjustCoords(box_points, original_size, target_length);

		transformed_boxes.push_back({ std::get<0>(transformed_box[0]), std::get<1>(transformed_box[0]),
									  std::get<0>(transformed_box[1]), std::get<1>(transformed_box[1]) });
	}

	return transformed_boxes;
}