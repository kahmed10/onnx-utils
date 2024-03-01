from extract import extract_model
import argparse
import os
import onnx
import sys
import onnx.utils
from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="get first N layers from an onnx model")
    parser.add_argument('--input-path', type=str,
                        dest='input_path', help='path to input onnx file', required=True)
    parser.add_argument('--output-path', type=str,
                        dest='output_path', help='path to output onnx file', required=True)
    parser.add_argument('--num-layers', type=int,  dest='num_layers',
                        help='Number of layers to chop from the top', required=True)
    parser.add_argument('--external-data', dest='external_data', default=False,
                        action='store_true', help='Whether the model has any external data or not')
    args = parser.parse_args()
    return args, parser


if __name__ == "__main__":
    args, parser = parse_args()
    onnx_model = args.input_path
    if not os.path.exists(onnx_model):
        raise ValueError(f"Invalid input model path: {onnx_model}")

    output_path = args.output_path
    num_layers = args.num_layers
    external_data = args.external_data
    onnx.checker.check_model(onnx_model)
    model = onnx.load(onnx_model, load_external_data=False)
    model = onnx.shape_inference.infer_shapes(model)
    if args.external_data:
        load_external_data_for_model(model, os.path.dirname(onnx_model))
    model_input_names = [i.name for i in model.graph.input]
    model_output_names = [i.name for i in model.graph.output]
    model_weights_names = [i.name for i in model.graph.initializer]
    export_input_names = []
    export_output_names = []
    for node in model.graph.node[0:num_layers]:
        for i in node.input:
            if (i in model_input_names and i not in export_input_names):
                export_input_names.append(i)
            if i in export_output_names:
                export_output_names.remove(i)
        for o in node.output:
            if o not in export_output_names and o not in model_weights_names:
                export_output_names.append(o)
    print("model has total: " + str(len(model.graph.node)) +
          " layers and out of that extracting the first " + str(num_layers) + " layers")
    print("inputs to the extracted models are: ", export_input_names)
    print("outputs of the extracted models are: ", export_output_names)

    extract_model(model, output_path,
                  export_input_names, export_output_names)

