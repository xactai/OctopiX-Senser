// TestApp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>
#include <unordered_map>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define TIME std::chrono::high_resolution_clock::now();

auto calculate_time = [](std::chrono::steady_clock::time_point start,
                         std::chrono::steady_clock::time_point end,
                         const std::string& message)
{
    double elapsed_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << message << elapsed_time << " msecs" << std::endl;
};

template <typename T> T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T> std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";

    for (int i = 0; i < v.size(); i++)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << "\n ";
        }
    }
    os << "]";
    return os;
}

std::unordered_map<std::string, std::string> args;

int main(int argc, char* argv[])
{
    // Can set to ORT_LOGGING_LEVEL_INFO or ORT_LOGGING_LEVEL_VERBOSE for more
    // info
    // Default running on NPU
    args["binary"] = argv[0];
    args["backend"] = "npu";
    args["image"] = "assets/images/cropped_1.png";
    args["model"] = "assets/models/quicksrnetsmall.qdq.onnx";
    args["tflite"] = "false";

    for (int i = 0; i < argc; i++)
    {
        if (!strcmp(argv[i],"--backend"))
        {
            args["backend"] = argv[i + 1];
            std::cout << "backend: " << args.at("backend") << std::endl;
        }
            
        if (!strcmp(argv[i], "--image"))
        {
            args["image"] = argv[i + 1];
            std::cout << "image: " << args.at("image") << std::endl;
        }
            
        if (!strcmp(argv[i], "--model"))
        {
            args["model"] = argv[i + 1];
            std::cout << "model: " << args.at("model") << std::endl;
        }

        if (!strcmp(argv[i], "--tflite"))
        {
            args["tflite"] = "true";
        }
    }

    auto model_load_start = TIME;

    const Ort::Env m_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "SuperResolution");

    std::string mp = args.at("model");

    std::wstring widestr = std::wstring(mp.begin(), mp.end());
    const wchar_t* model_path = widestr.c_str();

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    std::unordered_map<std::string, std::string> qnn_options;

    qnn_options["backend_path"] = args.at("backend") == "cpu" ? "QnnCpu.dll" : "QnnHtp.dll";

    // Additional options to set
    session_options.AppendExecutionProvider("QNN", qnn_options);

    Ort::Session session(m_env, model_path, session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputsNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    Ort::AllocatedStringPtr inputNames = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr outputNames = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> input_name{inputNames.get()};
    std::vector<const char*> output_name{outputNames.get()};

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    // load model end
    auto model_load_end = TIME;

    int inp_width = 0;
    int inp_height = 0;
    int inp_channels = 0;

    int out_width = 0;
    int out_height = 0;
    int out_channels = 0;

    if (args.at("tflite") == "true")
    {
        inp_width = static_cast<int>(inputDims.at(2));
        inp_height = static_cast<int>(inputDims.at(1));
        inp_channels = static_cast<int>(inputDims.at(3));

        out_width = static_cast<int>(outputDims.at(2));
        out_height = static_cast<int>(outputDims.at(1));
        out_channels = static_cast<int>(outputDims.at(3));
    }
    else
    {
        inp_width = static_cast<int>(inputDims.at(3));
        inp_height = static_cast<int>(inputDims.at(2));
        inp_channels = static_cast<int>(inputDims.at(1));

        out_width = static_cast<int>(outputDims.at(3));
        out_height = static_cast<int>(outputDims.at(2));
        out_channels = static_cast<int>(outputDims.at(1));
    }
    

    std::string image_path = args.at("image");

    cv::Mat input_image_bgr, input_image_rgb, resizedImageRGB, resizedImage, preprocessedImage;

    auto preprocessing_start = TIME;
    // preprocessing

    input_image_bgr = cv::imread(image_path, cv::ImreadModes::IMREAD_COLOR);

    cv::cvtColor(input_image_bgr, input_image_rgb, cv::ColorConversionCodes::COLOR_BGR2RGB);

    cv::resize(input_image_rgb, resizedImageRGB, cv::Size((int)inp_width, (int)inp_height));

    resizedImageRGB.convertTo(resizedImage, CV_32F, 1 / 255.0);

    //// HWC to CHW
     cv::dnn::blobFromImage(resizedImage, preprocessedImage);

     if (args.at("tflite") == "true") cv::transposeND(preprocessedImage, {0, 2, 3, 1}, preprocessedImage);


    size_t inputTensorSize = 1 * inp_channels * inp_height * inp_width;

    std::vector<float> inputTensorValues(inputTensorSize);
    std::copy(preprocessedImage.begin<float>(), preprocessedImage.end<float>(), inputTensorValues.begin());

    size_t outputTensorSize = 1 * out_channels * out_height * out_height;

    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize,
                                                           inputDims.data(), inputDims.size()));

    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(), outputTensorSize,
                                                            outputDims.data(), outputDims.size()));
    // End Preprocessing
    auto preprocessing_end = TIME;

    // Inference
    auto inference_start = TIME;
    Ort::RunOptions run_options{nullptr};

    session.Run(run_options, input_name.data(), inputTensors.data(), 1, output_name.data(), outputTensors.data(), 1);

    // End Inference
    auto inference_end = TIME;

    auto postprocessing_start = TIME;
    // Post Processing

    cv::Mat output_image = cv::Mat::zeros(out_width, out_height, CV_8UC3);

    cv::Mat output_image_r = cv::Mat::zeros(out_height, out_width, CV_32FC1);
    cv::Mat output_image_g = cv::Mat::zeros(out_height, out_width, CV_32FC1);
    cv::Mat output_image_b = cv::Mat::zeros(out_height, out_width, CV_32FC1);


    if (args.at("tflite") == "true")
    {
        // post processing for 1x512x512x3
        int index = 0;
        for (int y = 0; y < out_height; y++)
        {
            for (int x = 0; x < out_width; x++)
            {
                for (int c = 0; c < out_channels; c++)
                {
                    if (c == 0)
                        output_image_r.at<float>(y, x) = outputTensorValues[index];
                    if (c == 1)
                        output_image_g.at<float>(y, x) = outputTensorValues[index];
                    if (c == 2)
                        output_image_b.at<float>(y, x) = outputTensorValues[index];
                    index++;
                }
            }
        }

        // Fill the rawData with the RGB values from output_raw_image and saving it in output_image
        for (int x = 0; x < out_width; ++x)
        {
            for (int y = 0; y < out_height; ++y)
            {
                int r = static_cast<int>(output_image_r.at<float>(y, x) * 255.0f);
                int g = static_cast<int>(output_image_g.at<float>(y, x) * 255.0f);
                int b = static_cast<int>(output_image_b.at<float>(y, x) * 255.0f);
                output_image.at<cv::Vec3b>(y, x) = cv::Vec3b(r, g, b);
            }
        }
    }
    else
    {
        // post processing for 1x3x512x512
        int index = 0;
        for (int c = 0; c < out_channels; c++)
        {
            for (int y = 0; y < out_height; y++)
            {
                for (int x = 0; x < out_width; x++)
                {
                    if (c == 0)
                        output_image_r.at<float>(y, x) = outputTensorValues[index];
                    if (c == 1)
                        output_image_g.at<float>(y, x) = outputTensorValues[index];
                    if (c == 2)
                        output_image_b.at<float>(y, x) = outputTensorValues[index];
                    index++;
                }
            }
        }

        // Fill the rawData with the RGB values from output_raw_image and saving it in output_image
        for (int x = 0; x < out_width; ++x)
        {
            for (int y = 0; y < out_height; ++y)
            {
                int r = static_cast<int>(output_image_r.at<float>(y, x) * 255.0f);
                int g = static_cast<int>(output_image_g.at<float>(y, x) * 255.0f);
                int b = static_cast<int>(output_image_b.at<float>(y, x) * 255.0f);
                output_image.at<cv::Vec3b>(y, x) = cv::Vec3b(r, g, b);
            }
        }
    }

    cv::cvtColor(output_image, output_image, cv::ColorConversionCodes::COLOR_RGB2BGR);
    cv::imwrite("output.png", output_image);

    // End Post-Processing
    auto postprocessing_end = TIME;

    calculate_time(model_load_start, model_load_end, "Model Load Time: ");
    calculate_time(preprocessing_start, preprocessing_end, "Preprocessing Time: ");
    calculate_time(inference_start, inference_end, "Inference Time: ");
    calculate_time(postprocessing_start, postprocessing_end, "Postprocessing Time: ");
    calculate_time(preprocessing_start, postprocessing_end, "End-to-End Time: ");
}
