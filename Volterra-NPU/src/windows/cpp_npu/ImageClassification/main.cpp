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

void readLabels(std::string& labelFilePath, std::vector<std::string>& labels)
{
    std::string line;
    std::ifstream fp(labelFilePath);
    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
}

std::vector<float> inputTensorValues(size_t inputTensorSize)
{
    std::vector<float> local;
    local.resize(inputTensorSize);
    return local;
}

std::unordered_map<std::string, std::string> args;

int main(int argc, char* argv[])
{
    // Can set to ORT_LOGGING_LEVEL_INFO or ORT_LOGGING_LEVEL_VERBOSE for more
    // info
    // Default running on NPU
    args["binary"] = argv[0];
    args["backend"] = "npu";
    args["image"] = "assets/images/keyboard.jpg";
    args["model"] = "assets/models/inception_v3.qdq.onnx";

    for (int i = 0; i < argc; i++)
    {
        if (!strcmp("--backend", argv[i]))
        {
            args["backend"] = argv[i+1];
        }

        if (!strcmp("--image", argv[i]))
        {
            args["image"] = argv[i+1];
        }

        if (!strcmp("--model", argv[i]))
        {
            args["model"] = argv[i+1];
        }
    }

    auto model_load_start = TIME;
    // load model
    std::string labelFilePath = "assets/synset.txt";
    std::vector<std::string> labels;
    readLabels(labelFilePath, labels);

    const Ort::Env m_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Classification");

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

    std::cout << input_name.data() << std::endl;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    // load model end
    auto model_load_end = TIME;

    int inp_width = static_cast<int>(inputDims.at(3));
    int inp_height = static_cast<int>(inputDims.at(2));

    std::string image_path = args.at("image");

    cv::Mat input_image_bgr, input_image_rgb, resizedImageRGB, resizedImage, preprocessedImage;

    auto preprocessing_start = TIME;
    // preprocessing

    input_image_bgr = cv::imread(image_path, cv::ImreadModes::IMREAD_COLOR);

    cv::cvtColor(input_image_bgr, input_image_rgb, cv::ColorConversionCodes::COLOR_BGR2RGB);

    cv::resize(input_image_rgb, resizedImageRGB, cv::Size(inp_width, inp_height));

    resizedImageRGB.convertTo(resizedImage, CV_32F, 1 / 255.0);

    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;

    cv::merge(channels, 3, resizedImage);

    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    size_t inputTensorSize = 1 * 3 * inp_height * inp_width;

    std::vector<float> inputTensorValues(inputTensorSize);
    std::copy(preprocessedImage.begin<float>(), preprocessedImage.end<float>(), inputTensorValues.begin());

    size_t outputTensorSize = 1 * 1000;

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
    std::vector<int> predIds(1, 0);
    std::vector<std::string> predLabels(1);
    std::vector<float> confidences(1, 0.0f);

    float activation = 0;
    float maxActivation = std::numeric_limits<float>::lowest();
    float expSum = 0;

    for (int i = 0; i < labels.size(); i++)
    {
        activation = outputTensorValues.at(i);
        expSum += std::exp(activation);
        if (activation > maxActivation)
        {
            predIds.at(0) = i;
            maxActivation = activation;
        }
    }

    predLabels.at(0) = labels.at(predIds.at(0));
    confidences.at(0) = std::exp(maxActivation) / expSum;

    // End Post-Processing
    auto postprocessing_end = TIME;

    calculate_time(model_load_start, model_load_end, "Model Load Time: ");
    calculate_time(preprocessing_start, preprocessing_end, "Preprocessing Time: ");
    calculate_time(inference_start, inference_end, "Inference Time: ");
    calculate_time(postprocessing_start, postprocessing_end, "Postprocessing Time: ");
    calculate_time(preprocessing_start, postprocessing_end, "End-to-End Time: ");

    std::cout << "Prediction Label ID: " << predIds.at(0) << std::endl;
    std::cout << "Prediction Label: " << predLabels.at(0) << std::endl;
    std::cout << "Confidence: " << confidences.at(0) << std::endl;
}
