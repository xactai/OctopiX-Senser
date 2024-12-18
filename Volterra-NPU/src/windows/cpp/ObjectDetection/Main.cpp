// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#include <filesystem>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <chrono>

#include "ObjectDetectionApp.hpp"
#include "Utilities.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>


#define TIME std::chrono::high_resolution_clock::now();


namespace
{

constexpr uint32_t c_default_image_height = 640;
constexpr uint32_t c_default_image_width = 640;
constexpr App::BackendOption c_default_backend = App::BackendOption::Npu;

constexpr const char* c_option_backend = "--backend";
constexpr const char* c_option_output_image_path = "--output_image";
constexpr const char* c_option_qnn_options = "--qnn_options";
constexpr const char* c_option_precision = "--precision";
constexpr const char* c_option_image_path = "--image";
constexpr const char* c_option_model_path = "--model";
constexpr const char* c_option_input_image_height = "--model_input_ht";
constexpr const char* c_option_input_image_width = "--model_input_wt";

App::BackendOption GetBackendOptionFromCli(const char* cliOption)
{
    if (strcmp(cliOption, "cpu") == 0)
    {
        return App::BackendOption::Cpu;
    }
    if (strcmp(cliOption, "npu") == 0)
    {
        return App::BackendOption::Npu;
    }
    std::ostringstream err_msg;
    err_msg << "backend must be one of following: cpu or npu. Provided: " << cliOption << ".";
    throw std::runtime_error(err_msg.str());
}

/**
 * PrepareQNNOptions: Creates QNNOptions map from cli options
 *
 * Input CLI options template: "key1=val1;key2=val2;key3=val3;"
 * Output unordered map: { { "key1" : "val1" }, { "key2" : "val2" }, { "key3" :
 * "val3" } }
 *
 * @param cliOption qnn options string
 * @returns unordered_map<string, string>
 */
std::unordered_map<std::string, std::string> PrepareQNNOptions(const char* cliOption)
{
    std::unordered_map<std::string, std::string> qnn_options;
    const std::string qnn_options_str(cliOption);
    std::string map_delimiter(";");
    std::string kval_delimiter("=");

    size_t length = qnn_options_str.size();
    size_t s_index = 0;
    while (s_index < qnn_options_str.size() && s_index != std::string::npos)
    {
        // read key=val; pair and add into existing qnn_options map
        size_t key_index = qnn_options_str.find(kval_delimiter, s_index);
        std::string key = qnn_options_str.substr(s_index, key_index - s_index);

        key_index++;
        size_t val_index = qnn_options_str.find(map_delimiter, key_index);
        if (val_index == std::string::npos)
        {
            std::ostringstream err_msg;
            err_msg << "Incorrect qnn_options specified. Options must be of format "
                       "<key-1>=<val-1>;<key-2>=<val-2>;";
            throw std::runtime_error(err_msg.str());
        }
        std::string val = qnn_options_str.substr(key_index, val_index - key_index);
        qnn_options[key] = val;
        s_index = val_index + 1;
    }
    return qnn_options;
}

void PrintHelp()
{
    std::cout << "\nExample command line use:\n";
    std::cout << "./ObjectDetection.exe --backend npu " << c_option_model_path << " <model_path> "
              << c_option_image_path << " <input image path>\n";
    std::cout << "\n::::::::Object Detection App options::::::::\n";
    std::cout << "\nRequired options:\n\n";
    std::cout << c_option_model_path << " <local_path>: [Required] Path to local ONNX model.\n";
    std::cout << c_option_image_path
              << " <local_path>: [Required] Path to local input "
                 "image to run object detection on.\n";
    std::cout << "\nOptional options:\n\n";
    std::cout << "--backend <backend>: Default = npu. Set backend for inference. "
                 "Available options: cpu, npu.\n";
    std::cout << "--model_input_ht <model_input_ht>: Default = 640. Input "
                 "spatial height expected by model.\n";
    std::cout << "--model_input_wt <model_input_wt>: Default = 640. Input "
                 "spatial width expected by model.\n";
    std::cout << "--precision <precision>: Default = fp16. Set model precision. "
                 "Available options: fp32, fp16.\n";
    std::cout << c_option_output_image_path
              << " <local_path>: Default=''. If Set, writes "
                 "output image to provided local path.";
    std::cout << "--qnn_options <additional qnn options>: Default=''. Additional "
                 "qnn options to set.\n"
                 "This is dictionary option passed as a string and must follow "
                 "following template:\n"
                 "--qnn_options key1=val1;key2=val2;key3=val3;\n"
                 "Please refer to "
                 "https://onnxruntime.ai/docs/execution-providers/"
                 "QNN-ExecutionProvider.html#configuration-options for available "
                 "options.\n"
                 "NOTE: these options are not validated and are passed as a "
                 "dictionary to QNN Execution Provider.\n";
}

} // namespace

void count_time(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end, std::string message)
{
    double counter = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "\n*************************************************\n"
              << message << counter << "msecs"
              << "\n*************************************************\n";
}

auto print_status = [](bool status) -> std::string
{
    if (status)
    {
        return "Opened";
    }
    else
    {
        return "Not opened";
    }
};

int main(int argc, char* argv[])
{

    bool generate_ctx = false;
    App::Precision precision = App::Precision::Fp16;
    std::string backend;
    std::string model_path;
    std::string image_path;
    uint32_t input_image_height = c_default_image_height;
    uint32_t input_image_width = c_default_image_width;
    App::BackendOption backend_opt = c_default_backend;
    std::unordered_map<std::string, std::string> qnn_options;
    std::optional<std::string> output_image_path;

    // Arg parser
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], c_option_backend) == 0)
        {
            backend_opt = GetBackendOptionFromCli(argv[++i]);
        }
        else if (strcmp(argv[i], c_option_precision) == 0)
        {
            std::string option_precision(argv[++i]);
            if (option_precision == "fp32")
            {
                precision = App::Precision::Fp32;
            }
            else if (option_precision != "fp16")
            {
                std::cout << "--precision must be either fp32 or fp16.";
                return -1;
            }
        }
        else if (strcmp(argv[i], c_option_model_path) == 0)
        {
            model_path = argv[++i];
        }
        else if (strcmp(argv[i], c_option_image_path) == 0)
        {
            image_path = argv[++i];
        }
        else if (strcmp(argv[i], c_option_input_image_height) == 0)
        {
            input_image_height = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], c_option_input_image_width) == 0)
        {
            input_image_width = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], c_option_output_image_path) == 0)
        {
            output_image_path = argv[++i];
        }
        else if (strcmp(argv[i], c_option_qnn_options) == 0)
        {
            qnn_options = PrepareQNNOptions(argv[++i]);
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            PrintHelp();
            return 0;
        }
        else
        {
            std::cout << "Unsupported option " << argv[i] << " provided.\n";
            PrintHelp();
            return 1;
        }
    }

    // model_path and image_path must be provided
    if (model_path.empty() || image_path.empty())
    {
        std::cout << c_option_model_path << " and " << c_option_image_path << " must be provided.\n";
        PrintHelp();
        return 1;
    }

    try
    {
        App::ObjectDetectionApp app(model_path, input_image_height, input_image_width);

        auto start = TIME;
        // Prepare model
        app.PrepareModelForInference(backend_opt, precision, qnn_options);
        auto end = TIME;
        count_time(start, end, "Model prepare time: ");


        uint16_t frame_count = 1;
        std::string out_image_path = "output_image/";
        std::string output_window = "Object Detection";
        cv::Mat image;

        if (strcmp(image_path.c_str(), "camera") == 0)
        {
            cv::VideoCapture cap(0);
            if (cap.isOpened())
            {
                std::cout << "Webcam connected and reading images from webcam" << std::endl;

                while (true)
                {
                    bool captured = cap.read(image);

                    if (!captured)
                    {
                        std::cout << "End of Stream!" << std::endl;
                    }
                    // image = cv::imread(image_path, cv::IMREAD_COLOR);

                    cv::flip(image, image, 1);

                    std::string string_frame_count = std::to_string(frame_count);
                    string_frame_count = string_frame_count + ".jpg";

                    /*
                        Comment out to save the output images. It is to write the output images.
                    */

                    // std::string current_image_path = out_image_path + string_frame_count;
                    // cv::imwrite(current_image_path, image);

                    if (frame_count == 1)
                    {
                        start = TIME;
                        app.LoadInputImage(image);
                        end = TIME;
                        count_time(start, end, "Image loading time: ");

                        start = TIME;
                        app.RunInference();
                        end = TIME;
                        count_time(start, end, "Inference time: ");

                        app.ProcessOutputImage(image);
                        end = TIME;
                        count_time(start, end, "End-to-End Prediction Time: ");
                    }
                    else
                    {
                        //// Loading the image from saved paths
                        app.LoadInputImage(image);

                        app.RunInference();

                        app.ProcessOutputImage(image);
                    }

                    cv::namedWindow(output_window, cv::WINDOW_NORMAL);

                    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

                    cv::imshow(output_window, image);

                    uint8_t key = cv::waitKey(1);

                    frame_count++;

                    if (key == 27)
                    {
                        break;
                    }
                }
            }        
        }
        else
        {
             image = cv::imread(image_path, cv::IMREAD_COLOR);
             ////Load and cache inputs
             start = TIME;
             app.LoadInputs(image_path);
             end = TIME;
             count_time(start, end, "Image loading time: ");

             ////Run inference
             start = TIME;
             app.RunInference();
             end = TIME;
             count_time(start, end, "Inference time: ");

             ////Process output and show results
             start = TIME;
             app.ProcessOutput(image_path, output_image_path);
             end = TIME;
             count_time(start, end, "End-to-End Prediction Time: ");     
        
        }
        
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown error.\n";
        return 1;
    }
    return 0;
}
