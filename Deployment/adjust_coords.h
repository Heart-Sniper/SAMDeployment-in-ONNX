# include <iostream>
# include <algorithm>
# include <onnxruntime_cxx_api.h>
# include <assert.h>
# include <tuple>
# include <cmath>

std::tuple<int, int> CalculateResizeDim(int old_h, int old_w, int longside_length);

std::vector<std::tuple<float, float>> AdjustCoords(const std::vector<std::tuple<float, float>>& coords,
	const std::tuple<int, int>& original_size,
	int target_length);

std::vector<std::vector<float>> AdjustBox(const std::vector<std::vector<float>>& boxes,
	const std::tuple<int, int>& original_size,
	int target_length);
