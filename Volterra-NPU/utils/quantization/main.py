from custom_data_reader import CustomDataReader
from onnxruntime.quantization import CalibrationMethod
from onnxruntime.quantization import QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="xlsr", required=False, help="model name without extension")
parser.add_argument("--from_tflite", action="store_true", required=False, help="use it when tflite model was converted into onnx.")
parser.add_argument("--remove_activations", action="store_true", required=False, help="use it when tflite model was converted into onnx.")


args = parser.parse_args()

if __name__ == "__main__":
    model_name = args.model_name
    from_tflite = False
    remove_activations = True
    if args.from_tflite:
        from_tflite = True
    if args.remove_activations:
        remove_activations = True
    input_model_path = f"D:\Projects\Project_Volterra\model\{model_name}.onnx"
    
    output_model_path = f"D:\Projects\Project_Volterra\model\{model_name}.qdq.onnx"
    
    # my_data_reader = data_reader.DataReader(input_model_path)
    my_data_reader = CustomDataReader(calibration_image_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_data_reader"), 
                                      model_path=input_model_path, from_tflite=from_tflite)

    # Pre-process the original float32 model.
    preproc_model_path = f"D:\Projects\Project_Volterra\model\{model_name}.preproc.onnx"

    model_changed = qnn_preprocess_model(input_model_path, preproc_model_path)

    if model_changed:
        model_to_quantize = preproc_model_path
        print("Model changed with preprocessing")
    else:
        model_to_quantize = input_model_path
        print("Using default model")

    # Generate a suitable quantization configuration for this model.
    # Note that we're choosing to use uint16 activations and uint8 weights.
    qnn_config = get_qnn_qdq_config(model_to_quantize,
                                    my_data_reader,
                                    calibrate_method=CalibrationMethod.MinMax,
                                    activation_type=QuantType.QUInt16,  # uint16 activations
                                    weight_type=QuantType.QUInt8,       # uint8 weights
                                    keep_removable_activations=False if remove_activations else True)

    # Quantize the model.
    quantize(model_to_quantize, output_model_path, qnn_config)

