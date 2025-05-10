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
#include <sstream>
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
#include "color_codes.hpp"


#define TIME std::chrono::high_resolution_clock::now();

auto calculate_time = [](std::chrono::steady_clock::time_point start,
                         std::chrono::steady_clock::time_point end,
                         const std::string& message)
{
    double elapsed_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << message << elapsed_time << "msecs" << std::endl;
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

void readLabels(std::string& labelFilePath, std::vector<std::string>& labels, std::unordered_map<std::string, int>& label_index)
{
    std::string line;
    std::ifstream fp(labelFilePath);
    int index = 0;
    while (std::getline(fp, line))
    {
        labels.push_back(line);
        label_index[line] = index;
        index++;
    }
}

typedef struct
{
    int x;
    int y;
    int w;
    int h;
} BBox;

std::unordered_map<std::string, std::string> args;

int main(int argc, char* argv[])
{
    // Can set to ORT_LOGGING_LEVEL_INFO or ORT_LOGGING_LEVEL_VERBOSE for more
    // info
    // Bydefault running on NPU.
    args["binary"] = argv[0];
    args["backend"] = "npu";
    args["image"] = "assets/images/keyboard.jpg";
    args["model"] = "assets/models/yolov8_det.qdq.onnx";

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
    std::unordered_map<std::string, int> label_index;
    readLabels(labelFilePath, labels, label_index);

    const Ort::Env m_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "ObjectDetection");

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
    std::vector<const char*> input_name{inputNames.get()};

    Ort::AllocatedStringPtr boxesName = session.GetOutputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr scoresName = session.GetOutputNameAllocated(1, allocator);
    Ort::AllocatedStringPtr classidsName = session.GetOutputNameAllocated(2, allocator);
    std::vector<const char*> output_name{boxesName.get(), scoresName.get(), classidsName.get()};

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

    std::vector < std::vector<int64_t>> outputDims;

    for (int i = 0; i < numOutputNodes; i++)
    {
        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        outputDims.push_back(outputTensorInfo.GetShape());
    }

    // load model end
    auto model_load_end = TIME;

    int inp_width = static_cast<int>(inputDims.at(3));
    int inp_height = static_cast<int>(inputDims.at(2));
    int inp_channels = static_cast<int>(inputDims.at(1));

    std::string image_path = args.at("image");

    cv::Mat input_image_bgr, input_image_rgb, resizedImageRGB, resizedImage, preprocessedImage;

    auto preprocessing_start = TIME;
    // preprocessing

    input_image_bgr = cv::imread(image_path, cv::ImreadModes::IMREAD_COLOR);

    int image_height = input_image_bgr.rows;
    int image_width = input_image_bgr.cols;

    cv::Mat output_image_bgr(input_image_bgr);

    cv::cvtColor(input_image_bgr, input_image_rgb, cv::ColorConversionCodes::COLOR_BGR2RGB);

    cv::resize(input_image_rgb, resizedImageRGB, cv::Size(inp_width, inp_height));

    resizedImageRGB.convertTo(resizedImage, CV_32F, 1 / 255.0);

    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    size_t inputTensorSize = 1 * inp_channels * inp_height * inp_width;

    std::vector<float> inputTensorValues(inputTensorSize);
    std::copy(preprocessedImage.begin<float>(), preprocessedImage.end<float>(), inputTensorValues.begin());

    size_t outputBoxesSize = 1 * 8400 * 4;
    size_t outputScoresSize = 1 * 8400;
    size_t outputClassIdsSize = 1 * 8400;

    std::vector<float> outputBoxesValues(outputBoxesSize);
    std::vector<float> outputScoresValues(outputScoresSize);
    std::vector<float> outputClassIdsValues(outputClassIdsSize);

    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize,
                                                           inputDims.data(), inputDims.size()));

    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputBoxesValues.data(), outputBoxesSize,
                                                            outputDims[0].data(), outputDims[0].size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputScoresValues.data(), outputScoresSize,
                                                            outputDims[1].data(), outputDims[1].size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputClassIdsValues.data(), outputClassIdsSize,
                                                            outputDims[2].data(), outputDims[2].size()));
    // End Preprocessing
    auto preprocessing_end = TIME;

    // Inference
    auto inference_start = TIME;
    Ort::RunOptions run_options{nullptr};

    session.Run(run_options, input_name.data(), inputTensors.data(), numInputsNodes, output_name.data(), outputTensors.data(), numOutputNodes);

    // End Inference
    auto inference_end = TIME;

    // Post-processing
    auto postprocessing_start = TIME;

    auto output_coords = outputTensors[0].GetTensorData<float>();
    auto output_prob = outputTensors[1].GetTensorData<float>();
    auto output_class = outputTensors[2].GetTensorData<float>();

    std::vector<float> output_probs;

    for (int i = 0; i < 8400; i++)
    {
        output_probs.push_back(output_prob[i]);
    }

    cv::dnn::dnn4_v20230620::MatShape indices;

    std::vector<cv::Rect2d> bbox;

    int j = 0;
    for (int i = 0; i < 8400; i++)
    {
        j = i * 4;
        cv::Rect2d temp;
        int x1 = static_cast<int>(output_coords[j]);
        int y1 = static_cast<int>(output_coords[j+1]);
        int x2 = static_cast<int>(output_coords[j+2]);
        int y2 = static_cast<int>(output_coords[j+3]);
        temp.x = x1;
        temp.y = y1;
        temp.width = x2 - x1;
        temp.height = y2 - y1;
        bbox.push_back(temp);
    }

    float threshold_score = 0.05f;
    float nms_threshold = 0.3f;

    cv::dnn::NMSBoxes(bbox, output_probs, threshold_score, nms_threshold, indices);


    float ratio_h = static_cast<float>(image_height) / static_cast<float>(inp_height);
    float ratio_w = static_cast<float>(image_width) / static_cast<float>(inp_width);

    for (int i = 0; i < indices.size(); i++)
    {
        int x1 = static_cast<int>(bbox[static_cast<int>(indices.at(i))].x * ratio_w);
        int y1 = static_cast<int>(bbox[static_cast<int>(indices.at(i))].y * ratio_h);
        int x2 = static_cast<int>((bbox[static_cast<int>(indices.at(i))].width + bbox[static_cast<int>(indices.at(i))].x) * ratio_w);
        int y2 = static_cast<int>((bbox[static_cast<int>(indices.at(i))].height + bbox[static_cast<int>(indices.at(i))].y) * ratio_h);

        std::string label = labels[static_cast<int>(output_class[indices.at(i)])];

        std::ostringstream tag;
        tag << label << ": " << std::fixed << std::setprecision(5) << output_probs[indices.at(i)];

        int baseline = 0;

        cv::Size textSize = cv::getTextSize(tag.str(), cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);

        cv::rectangle(output_image_bgr, cv::Point(x1, y1), cv::Point(x2, y2), colorCodes[label_index.at(label)], 2);
        cv::rectangle(output_image_bgr, cv::Point(x1, y1 - textSize.height - static_cast<int>((0.02 * image_height))),
                      cv::Point(x1 + textSize.width + static_cast<int>((0.01 * image_width)), y1), colorCodes[label_index.at(label)], -1);
        cv::putText(output_image_bgr, tag.str(), cv::Point(x1, y1- static_cast<int>((0.01*image_height))), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
        cv::imwrite("output.png", output_image_bgr);
    }

    // End Post-Processing
    auto postprocessing_end = TIME;

    calculate_time(model_load_start, model_load_end, "Model Load Time: ");
    calculate_time(preprocessing_start, preprocessing_end, "Preprocessing Time: ");
    calculate_time(inference_start, inference_end, "Inference Time: ");
    calculate_time(postprocessing_start, postprocessing_end, "Postprocessing Time: ");
    calculate_time(preprocessing_start, postprocessing_end, "End-to-End Time: ");
}
