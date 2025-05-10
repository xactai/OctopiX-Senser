# I am trying to use the onnx model from this repository with Python3.11 and onnxruntime-qnn=1.19.0 on Windows Platform on Snapdragon 8cx Gen 3 Processor (Windows Dev kit 2023). But during the inference I am getting this error:
# `FAIL : Node 'Conv' OpType:Conv with domain:com.ms.internal.nhwc was inserted using the NHWC format as requested by QNNExecutionProvider, but was not selected by that EP. This means the graph is now invalid as there will not be an EP able to run the node. This could be a bug in layout transformer, or in the GetCapability implementation of the EP.`

# # Code:
# options.add_session_config_entry("session.disable_cpu_ep_fallback", "0")

# options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# self.session = onnxruntime.InferenceSession(path,
#                                             sess_options=options,
#                                             providers=["QNNExecutionProvider"],
#                                             provider_options=[{"backend_path": "QnnHtp.dll")

# outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

# # Environment:
# `
# QNN_SDK_ROOT=C:\Qualcomm\AIStack\QAIRT\2.22.0.240425
# `

# # Error:
# `
# unknown ARM CPU part 0xd4b ignored

# unknown ARM CPU part 0xd4b ignored

# unknown ARM CPU part 0xd4b ignored

# unknown ARM CPU part 0xd4b ignored

# unknown ARM CPU part 0xd4c ignored

# unknown ARM CPU part 0xd4c ignored

# unknown ARM CPU part 0xd4c ignored

# unknown ARM CPU part 0xd4c ignored

# 2024-11-29 13:12:11.8863606 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `/model/cnn/1/Relu` of type `Relu` with error code 3110

# 2024-11-29 13:12:11.8919005 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `/model/cnn/3/net/1/Relu` of type `Relu` with error code 3110

# 2024-11-29 13:12:11.8972361 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `/model/cnn/4/net/1/Relu` of type `Relu` with error code 3110

# 2024-11-29 13:12:11.9042892 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `/model/cnn/5/net/1/Relu` of type `Relu` with error code 3110

# 2024-11-29 13:12:11.9111319 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `/model/tail/1/Relu` of type `Relu` with error code 3110

# 2024-11-29 13:12:11.9166808 [W:onnxruntime:, qnn_execution_provider.cc:664 onnxruntime::QNNExecutionProvider::GetCapability] QNN EP does not support a single Quantize/Dequantize node in a partition.
# 2024-11-29 13:12:11.9239293 [E:onnxruntime:, qnn_model_wrapper.cc:138 onnxruntime::qnn::QnnModelWrapper::CreateQnnInputOutputTensors] Input name not exist: model.cnn.0.weight_dq
# 2024-11-29 13:12:11.9291546 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `Conv` of type `Conv2d` with error code 3110

# 2024-11-29 13:12:11.9339917 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `/model/cnn/1/Relu` of type `Relu` with error code 3110

# 2024-11-29 13:12:11.9387543 [E:onnxruntime:, qnn_model_wrapper.cc:138 onnxruntime::qnn::QnnModelWrapper::CreateQnnInputOutputTensors] Input name not exist: model.cnn.2.weight_dq
# 2024-11-29 13:12:11.9431608 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `Conv_token_6` of type `Conv2d` with error code 3110

# 2024-11-29 13:12:11.9490821 [E:onnxruntime:, qnn_model_wrapper.cc:138 onnxruntime::qnn::QnnModelWrapper::CreateQnnInputOutputTensors] Input name not exist: model.cnn.3.net.0.weight_dq
# 2024-11-29 13:12:11.9528103 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `Conv_token_12` of type `Conv2d` with error code 3110

# 2024-11-29 13:12:11.9584756 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `/model/cnn/3/net/1/Relu` of type `Relu` with error code 3110

# 2024-11-29 13:12:11.9640692 [E:onnxruntime:, qnn_model_wrapper.cc:138 onnxruntime::qnn::QnnModelWrapper::CreateQnnInputOutputTensors] Input name not exist: model.cnn.3.net.2.weight_dq
# 2024-11-29 13:12:11.9677988 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `Conv_token_18` of type `Conv2d` with error code 3110

# 2024-11-29 13:12:11.9724161 [E:onnxruntime:, qnn_model_wrapper.cc:138 onnxruntime::qnn::QnnModelWrapper::CreateQnnInputOutputTensors] Input name not exist: model.cnn.4.net.0.weight_dq
# 2024-11-29 13:12:11.9771572 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `Conv_token_24` of type `Conv2d` with error code 3110

# 2024-11-29 13:12:11.9818775 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `/model/cnn/4/net/1/Relu` of type `Relu` with error code 3110

# 2024-11-29 13:12:11.9871576 [E:onnxruntime:, qnn_model_wrapper.cc:138 onnxruntime::qnn::QnnModelWrapper::CreateQnnInputOutputTensors] Input name not exist: model.cnn.4.net.2.weight_dq
# 2024-11-29 13:12:11.9915960 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `Conv_token_30` of type `Conv2d` with error code 3110

# 2024-11-29 13:12:11.9965127 [E:onnxruntime:, qnn_model_wrapper.cc:138 onnxruntime::qnn::QnnModelWrapper::CreateQnnInputOutputTensors] Input name not exist: model.cnn.5.net.0.weight_dq
# 2024-11-29 13:12:11.9999194 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `Conv_token_36` of type `Conv2d` with error code 3110

# 2024-11-29 13:12:12.0044381 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `/model/cnn/5/net/1/Relu` of type `Relu` with error code 3110

# 2024-11-29 13:12:12.0107981 [E:onnxruntime:, qnn_model_wrapper.cc:138 onnxruntime::qnn::QnnModelWrapper::CreateQnnInputOutputTensors] Input name not exist: model.cnn.5.net.2.weight_dq
# 2024-11-29 13:12:12.0146768 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `Conv_token_42` of type `Conv2d` with error code 3110

# 2024-11-29 13:12:12.0198233 [E:onnxruntime:, qnn_model_wrapper.cc:138 onnxruntime::qnn::QnnModelWrapper::CreateQnnInputOutputTensors] Input name not exist: model.residual.weight_dq
# 2024-11-29 13:12:12.0243246 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `Conv_token_48` of type `Conv2d` with error code 3110

# 2024-11-29 13:12:12.0296923 [E:onnxruntime:, qnn_model_wrapper.cc:138 onnxruntime::qnn::QnnModelWrapper::CreateQnnInputOutputTensors] Input name not exist: model.tail.0.weight_dq
# 2024-11-29 13:12:12.0332154 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `Conv_token_54` of type `Conv2d` with error code 3110

# 2024-11-29 13:12:12.0376411 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `/model/tail/1/Relu` of type `Relu` with error code 3110

# 2024-11-29 13:12:12.0447864 [E:onnxruntime:, qnn_model_wrapper.cc:138 onnxruntime::qnn::QnnModelWrapper::CreateQnnInputOutputTensors] Input name not exist: model.tail.2.weight_dq
# 2024-11-29 13:12:12.0497153 [W:onnxruntime:, qnn_model_wrapper.cc:244 onnxruntime::qnn::QnnModelWrapper::CreateQnnNode] QNN.backendValidateOpConfig() failed for node `Conv_token_60` of type `Conv2d` with error code 3110

# Traceback (most recent call last):
#   File "C:\volterra_research\refactored_code\python\super-resolution\super_resolution.py", line 54, in <module>
#     sr = SuperResolution(
#          ^^^^^^^^^^^^^^^^
#   File "C:\volterra_research\refactored_code\python\super-resolution\super_resolution\resolution.py", line 12, in __init__
#     self.initialize_model(path)
#   File "C:\volterra_research\refactored_code\python\super-resolution\super_resolution\resolution.py", line 24, in initialize_model
#     self.session = onnxruntime.InferenceSession(path,
#                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\volterra_research\volterravenv\Lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 419, in __init__
#     self._create_inference_session(providers, provider_options, disabled_optimizers)
#   File "C:\volterra_research\volterravenv\Lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 491, in _create_inference_session
#     sess.initialize_session(providers, provider_options, disabled_optimizers)
# onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Node 'Conv' OpType:Conv with domain:com.ms.internal.nhwc was inserted using the NHWC format as requested by QNNExecutionProvider, but was not selected by that EP. This means the graph is now invalid as there will not be an EP able to run the node. This could be a bug in layout transformer, or in the GetCapability implementation of the EP.
# `

# Kindly help me to resolve this issue.