#include <iostream>
#include <opencv2/opencv.hpp> 
#include <chrono>
#include <string>
#include <tuple>
#include <cmath>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <tensorrt_provider_factory.h>

#include"img_processing.h"
#include "adjust_coords.h"

static wchar_t* Char2Wchar(const char* ch);
std::vector<const char*> VecStr2Vecchar(const std::vector<std::string>& vec_str);

Ort::Session CreateSession(const std::string& model_path, Ort::Env& env, OrtCUDAProviderOptions cuda_options)
{
	// set session options
	Ort::SessionOptions session_options;
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	// open CUDA acceleration
	session_options.AppendExecutionProvider_CUDA(cuda_options);

	// load ONNX model & create a session
	wchar_t* model = Char2Wchar(model_path.c_str());
	Ort::Session session(env, model, session_options);

	session_options.release();
	free(model);

	return session;
}

// Function to get input and output information from the session
void GetSessionInfo(const Ort::Session& session,
	std::vector<std::string>& input_nodename,
	std::vector<std::string>& output_nodename) //string
{
	Ort::AllocatorWithDefaultOptions allocator;

	// Names and numbers of input nodes and output nodes in encoder
	size_t input_count = session.GetInputCount();
	size_t output_count = session.GetOutputCount();

	//std::vector<std::string> input_nodename;
	//std::vector<std::string> output_nodename;

	std::cout << "[Encoder] input node count: " << input_count << std::endl;
	std::cout << "          input node name: ";
	for (size_t i = 0; i < input_count; ++i)
	{
		auto nodename = session.GetInputNameAllocated(i, allocator);
		std::cout << nodename << " ";
		input_nodename.push_back("");
		input_nodename[i].append(nodename.get());
	}
	std::cout << std::endl;

	std::cout << "[Encoder] output node count: " << output_count << std::endl;
	std::cout << "          output node name: ";
	for (size_t i = 0; i < output_count; ++i)
	{
		auto nodename = session.GetOutputNameAllocated(i, allocator);
		std::cout << nodename << " ";
		output_nodename.push_back("");
		output_nodename[i].append(nodename.get());
	}
	std::cout << std::endl;
}


int main()
{
	//------------------------------------------------Customized parameters--------------------------------------------------------
	
	// path to image
	std::string img_path = "D:\\Project\\SAMDeployment\\InputData\\testdata\\wanderer2.jpg";

	// input mode
	// (please input "point" or "box")
	std::string mode = "box";

	// input coordinate

	// encoder model path & decoder model path
	std::string encoder_model_path = "D:\\Project\\WovenBagDetection\\GenerateMask\\SAM_ONNX\\sam_vit_b_VITencoder.onnx";
	std::string decoder_model_path = "D:\\Project\\WovenBagDetection\\GenerateMask\\SAM_ONNX\\sam_vit_b_pmdecoder.onnx";

	//-----------------------------------------------------Public Space------------------------------------------------------------
#pragma region PublicSpace

	// get basic information from input image
	cv::Mat image = cv::imread(img_path);
	std::tuple<int, int> img_size = std::make_tuple(image.rows, image.cols);

	// set environment
	Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "SAM");

	// set device
	OrtCUDAProviderOptions cuda_options;
	Ort::AllocatorWithDefaultOptions allocator;
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

#pragma endregion
	//#######################################################Encoder################################################################
	// 
	//-------------------------------------------------onnxruntime initialize-------------------------------------------------------
#pragma region EncoderInitialize

	// Create session for encoder
	Ort::Session encoder_session = CreateSession(encoder_model_path, env, cuda_options);

	std::vector<std::string> input_name_en, output_name_en;
	GetSessionInfo(encoder_session, input_name_en, output_name_en);
	std::vector<const char*> input_names_en, output_names_en;
	for (const auto& name : input_name_en) input_names_en.push_back(name.c_str());
	for (const auto& name : output_name_en) output_names_en.push_back(name.c_str());

#pragma endregion

	//-------------------------------------------------data preprocessing-----------------------------------------------------------

	cv::Mat img = image.clone();
	// get dimensions of inputs and outputs
	std::vector<int64_t> input_dim_en = encoder_session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	//std::vector<int64_t> output_dim_en = encoder_session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	img = ImgTransform(img, input_dim_en);
	std::vector<float> input_en = FlattenImageChannels(img, input_dim_en);

	//-------------------------------------------------------inference--------------------------------------------------------------
#pragma region EncoderInference

	// create input tensor
	std::vector<int64_t> input_shape_en;
	input_shape_en = encoder_session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

	std::vector<Ort::Value> input_tensor_en;
	input_tensor_en.push_back(Ort::Value::CreateTensor(memory_info,
													   input_en.data(), input_en.size(),
													   input_shape_en.data(), input_shape_en.size()));

	// run session for encoder
	std::cout << "[Encoder] Inference ON" << std::endl;
	std::chrono::steady_clock::time_point start_time_en = std::chrono::steady_clock::now();
	auto output_tensor_en = encoder_session.Run(Ort::RunOptions {nullptr},	  // run options
												input_names_en.data(),          // name of model input_en node
												input_tensor_en.data(),        // input_en tensors
												input_tensor_en.size(),        // number of input_en node
												output_names_en.data(),         // name of model output node
												output_names_en.size());        // number of output node

	std::chrono::steady_clock::time_point end_time_en = std::chrono::steady_clock::now();
	std::chrono::duration<double> time_en = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_en - start_time_en);
	std::cout << "          Inference Time Consumption: " << time_en.count() * 1000 / 100.0 << " ms" << std::endl;
	std::cout << "[Encoder] Inference OFF" << std::endl;

	// get output
	std::cout << "[Encoder] output tensor dimension: " << output_tensor_en.size() << std::endl;

	std::cout << "          output value dimension: ";
	for (Ort::Value& ele : output_tensor_en)
	{
		auto num_ele = ele.GetTensorTypeAndShapeInfo().GetShape();
		for (auto i : num_ele)
		{
			std::cout << i << " ";
		}
	}
	std::cout << std::endl;

#pragma endregion
	//-------------------------------------------------------release----------------------------------------------------------------
	 
	encoder_session.release();


#pragma endregion
	std::cout << std::endl << "-----Encoder ended---Decoder started-----" << std::endl << std::endl;
	//########################################################Decoder###############################################################
	// The inputs to the decoder are fixed to 6 items, in that order:
	// [0]image_embeddings; [1]point_coords; [2]point_labels; [3]mask_input; [4]has_mask_input; [5]orig_im_size
	// The name of the input node can be changed, but the order of the parameters is not changeable.
	//------------------------------------------------onnxruntime initialize--------------------------------------------------------
#pragma region DecoderInitialize

	Ort::Session decoder_session = CreateSession(decoder_model_path, env, cuda_options);

	// names and numbers of input nodes and output nodes
	size_t input_count_de = decoder_session.GetInputCount();
	size_t output_count_de = decoder_session.GetOutputCount();
	std::vector<std::string> input_nodename_de;
	std::vector<std::string> output_nodename_de;

	std::cout << "[Decoder] input node count " << input_count_de << std::endl;
	std::cout << "          input node name: ";
	for (auto i = 0; i < input_count_de; ++i)
	{
		auto nodename = decoder_session.GetInputNameAllocated(i, allocator);
		std::cout << nodename << " ";

		input_nodename_de.push_back("");
		input_nodename_de[i].append(nodename.get());
	}

	std::cout << std::endl << "[Decoder] output node count " << output_count_de << std::endl;
	std::cout << "          output node name: ";
	for (auto i = 0; i < output_count_de; ++i)
	{
		auto nodename = decoder_session.GetOutputNameAllocated(i, allocator);
		std::cout << nodename << " ";

		output_nodename_de.push_back("");
		output_nodename_de[i].append(nodename.get());
	}
	std::cout << std::endl;

	std::vector<const char*> input_name_de;
	input_name_de = VecStr2Vecchar(input_nodename_de);
	std::vector<const char*> output_name_de;
	output_name_de = VecStr2Vecchar(output_nodename_de);

#pragma endregion
	//-----------------------------------------------------data preprocessing-------------------------------------------------------

	// get image embeddings from encoder output tensor [1, 256, 64, 64]
	std::vector<float> img_embedding_value;
	for (const Ort::Value& ele : output_tensor_en)
	{
		const float* float_data = ele.GetTensorData<float>();
		size_t num_ele = ele.GetTensorTypeAndShapeInfo().GetElementCount();

		img_embedding_value.insert(img_embedding_value.end(), float_data, float_data + num_ele);
	}

	std::vector<int64_t> img_embedding_shape;
	img_embedding_shape = decoder_session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

	Ort::Value img_embedding = Ort::Value::CreateTensor(memory_info,
													    img_embedding_value.data(), img_embedding_value.size(),
													    img_embedding_shape.data(), img_embedding_shape.size());

	//- - - - - - - - - - - - - - - - - - - - - - - - - - - point input- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#if 0	 
	// get point coordinates 

	//MouseEvent action = CaptureMouseAction(image);
	//std::tuple<float, float> point = { static_cast<float>(action.x), static_cast<float>(action.y) };
	float x, y;    // test point
	x = 323; y = 541;
	std::tuple<float, float> point{x, y};
	std::vector<std::tuple<float, float>> inputpoints_list;
	inputpoints_list.push_back(point);

	int target_length = std::max(input_dim_en[2], input_dim_en[3]);    // [1024, 1024]

	std::vector<std::tuple<float, float>> points_list;
	points_list = AdjustCoords(inputpoints_list, img_size, target_length);
	int points_num = points_list.size();
	std::cout << "[Decoder] number of input points: " << points_list.size() << std::endl;

	std::vector<float> points_coord_value;
	for (std::tuple<float, float> p : points_list)
	{
		points_coord_value.push_back(std::get<0>(p));
		points_coord_value.push_back(std::get<1>(p));
	}
	std::vector<int64_t> points_coord_shape;
	points_coord_shape = session_de.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
	points_coord_shape[1] = points_num;

	Ort::Value points_coord = Ort::Value::CreateTensor(memory_info,
													   points_coord_value.data(), points_coord_value.size(),
													   points_coord_shape.data(), points_coord_shape.size());

	// get point labels
	// label == 1 : foreground
	// label == 0 : background
	// assuming label = 1 as test
	int points_label_num = points_num;
	std::vector<float> points_label_value;
	for (auto i = 0; i < points_label_num; ++i)
	{
		points_label_value.push_back(1);
	}

	std::vector<int64_t> points_label_shape;
	points_label_shape = { 1, points_label_num };

	Ort::Value points_label = Ort::Value::CreateTensor(memory_info,
													   points_label_value.data(), points_label_value.size(),
													   points_label_shape.data(), points_label_shape.size());
#endif
	//- - - - - - - - - - - - - - - - - - - - - - - - - - - box input- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#if 1
	// test box:[484, 940, 517, 1003]
	std::vector<float> box_point(4);
	std::vector<std::vector<float>> boxes;

	box_point = { 478, 930, 521, 1014 };
	boxes.push_back(box_point);

	int target_length = std::max(input_dim_en[2], input_dim_en[3]);
	std::vector<std::vector<float>> boxes_list = AdjustBox(boxes, img_size, target_length);
	
	std::vector<float> boxes_point_coords_value;
	for (std::vector<float> box : boxes_list)
	{
		boxes_point_coords_value.push_back(box[0]);
		boxes_point_coords_value.push_back(box[1]);
		boxes_point_coords_value.push_back(box[2]);
		boxes_point_coords_value.push_back(box[3]);
	}

	std::vector<int64_t> boxes_points_coord_shape;    // [1, 2, 2]
	boxes_points_coord_shape = decoder_session.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
	int boxes_num = boxes_list.size();
	boxes_points_coord_shape[1] = boxes_num * 2;
	//boxes_points_coord_shape = { 1, 2, 2 };
	
	Ort::Value points_coord = Ort::Value::CreateTensor(memory_info,
													   boxes_point_coords_value.data(), boxes_point_coords_value.size(),
													   boxes_points_coord_shape.data(), boxes_points_coord_shape.size());
#endif

#if 1
	// get labels of boxes
	// label == 1 : foreground
	// label == 0 : background
	// assuming label = 1 as test
	std::vector<float> boxes_points_label_value;
	int boxes_label_num = boxes_num;
	for (auto i = 0; i < boxes_label_num; ++i)
	{
		boxes_points_label_value.push_back(2);
		boxes_points_label_value.push_back(3);
	}

	std::vector<int64_t> boxes_points_label_shape;
	boxes_points_label_shape = { 1, boxes_label_num * 2 };

	Ort::Value points_label = Ort::Value::CreateTensor(memory_info,
													   boxes_points_label_value.data(), boxes_points_label_value.size(),
													   boxes_points_label_shape.data(), boxes_points_label_shape.size());

#endif

	// get mask and mask holder
	// A low resolution mask input to the model, typically coming from a previous prediction iteration. 
	// Has form 1xHxW, where for SAM, H = W = 256.
	std::vector<float> mask_value;
	for (int i = 0; i < 1 * 1 * 256 * 256; ++i)
	{
		mask_value.push_back(0.0f);
	}
	std::vector<int64_t> mask_shape = decoder_session.GetInputTypeInfo(3).GetTensorTypeAndShapeInfo().GetShape();
	Ort::Value mask_input = Ort::Value::CreateTensor(memory_info,
													 mask_value.data(), mask_value.size(),
													 mask_shape.data(), mask_shape.size());

	std::vector<float> mask_holder_value(1, 0.0f);
	std::vector<int64_t> mask_holder_shape = decoder_session.GetInputTypeInfo(4).GetTensorTypeAndShapeInfo().GetShape();
	Ort::Value mask_holder = Ort::Value::CreateTensor(memory_info,
													  mask_holder_value.data(), mask_holder_value.size(),
													  mask_holder_shape.data(), mask_holder_shape.size());

	// get origin size from input image
	std::vector<float> orig_im_size_value;
	orig_im_size_value.push_back(image.rows);
	orig_im_size_value.push_back(image.cols);

	std::vector<int64_t> orig_im_size_shape;
	orig_im_size_shape = decoder_session.GetInputTypeInfo(5).GetTensorTypeAndShapeInfo().GetShape();

	Ort::Value orig_im_size = Ort::Value::CreateTensor(memory_info,
													   orig_im_size_value.data(), orig_im_size_value.size(),
													   orig_im_size_shape.data(), orig_im_size_shape.size());

	//--------------------------------------------------------inference-------------------------------------------------------------

	// create input tensor from output of encoder
	std::vector<Ort::Value> input_tensor_de;
	input_tensor_de.push_back(std::move(img_embedding));
	input_tensor_de.push_back(std::move(points_coord));
	input_tensor_de.push_back(std::move(points_label));
	input_tensor_de.push_back(std::move(mask_input));
	input_tensor_de.push_back(std::move(mask_holder));
	input_tensor_de.push_back(std::move(orig_im_size));

	// run session for decoder
	std::cout << "[Decoder] Inference ON" << std::endl;
	std::chrono::steady_clock::time_point start_time_de = std::chrono::steady_clock::now();
	auto output_tensor_de = decoder_session.Run(Ort::RunOptions {nullptr},		// run options
										   input_name_de.data(),            // name of model input node
										   input_tensor_de.data(),          // input_en tensors
										   input_tensor_de.size(),          // number of input node
										   output_name_de.data(),           // name of model output node
										   output_name_de.size());          // number of output node
	std::chrono::steady_clock::time_point end_time_de = std::chrono::steady_clock::now();
	std::chrono::duration<double> time_de = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_de - start_time_de);
	std::cout << "          Time Consumption: " << time_de.count() * 1000 / 100.0 << " ms" << std::endl;
	std::cout << "[Decoder] Inference OFF" << std::endl;

	// check out dimensions of outputs
	std::cout << "[Decoder] output dimensions: " << std::endl;
	int i = 0;
	for (const Ort::Value& output_value : output_tensor_de)
	{
		Ort::TensorTypeAndShapeInfo type_info = output_value.GetTensorTypeAndShapeInfo();
		std::vector<int64_t> shape = type_info.GetShape();
		auto nodename = decoder_session.GetOutputNameAllocated(i, allocator);
		std::cout << "          " << nodename.get() << ": [ ";
		for (int64_t dimension : shape)
		{
			std::cout << dimension << " ";
		}
		std::cout << "]" << std::endl;
		++i;
	}
	//-------------------------------------------------------release----------------------------------------------------------------
	
	//session_de.release();
	// 
	//-----------------------------------------------------image display------------------------------------------------------------

	int mask_width = 0, mask_height = 0;
	float mask_threshold = 0.0f;
	std::vector<float> mask;

	for (size_t i = 0; i < output_tensor_de.size(); ++i)
	{
		auto nodename = decoder_session.GetOutputNameAllocated(i, allocator);
		const std::string name = nodename.get();
		if (name == "masks")
		{
			mask_height = output_tensor_de[i].GetTensorTypeAndShapeInfo().GetShape()[2];
			mask_width = output_tensor_de[i].GetTensorTypeAndShapeInfo().GetShape()[3];
			int num_ele = mask_width * mask_height;
			mask.resize(num_ele);

			float* data = output_tensor_de[i].GetTensorMutableData<float>();

			for (int j = 0, k = 0; j < num_ele; ++j, ++k)
			{
				mask[j] = (data[k] > mask_threshold ? 255 : 0);
			}
		}
	}

	cv::Mat mask_img(mask_height, mask_width, CV_32FC1, mask.data());

	mask_img.convertTo(mask_img, CV_8UC1);

	std::vector<cv::Mat> mv;
	cv::split(image, mv);

	cv::bitwise_or(mv[2], mask_img, mv[2]);

	cv::Mat show_img;
	cv::merge(mv, show_img);

	cv::imshow("Image With Mask", show_img);
	cv::waitKey(0);

	return 0;

}



/**
 * Function: Char2Wchar
 * Description: Converts a C-style string from char to wchar_t and allocates memory for the new wide string.
 *				Called in 'CreateSession' function, and memory is freed in it.
 *
 * @param ch (const char*): The input C-style string (char).
 *
 * @return (wchar_t*): A dynamically allocated wide string (wchar_t*) with the converted content.                     
 */
static wchar_t* Char2Wchar(const char* ch)
{
	const size_t len = strlen(ch) + 1;
	wchar_t* wch = (wchar_t*)malloc(len * sizeof(wchar_t));

	size_t convert = 0;
	mbstowcs_s(&convert, wch, len, ch, _TRUNCATE);

	return wch;
}

/****
* Function: VecStr2Vecchar
* Description: Converts a vector of C++ strings to a vector of const char*.
* 
* @param vec_str (const std::vector<std::string>&): The input vector of strings.
* 
* @return (std::vector<const char*>): A vector of const char* containing the converted strings.
****/
std::vector<const char*> VecStr2Vecchar(const std::vector<std::string>& vec_str)
{
	std::vector<const char*> vec_char;
	vec_char.reserve(vec_str.size());
	for (const std::string& i : vec_str)
	{
		vec_char.push_back(i.c_str());
	}

	return vec_char;
}
