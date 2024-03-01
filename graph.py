from extract import extract_model
import os
import onnx
import sys
import onnx.utils
from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model

if len(sys.argv) != 4:
    print(
        "Usage: python graph.py [/path/to/model.onnx] [path/to/subgraphx.onnx] [num_layers]")
    exit()

onnx_model = sys.argv[1]
subgraphx_path = sys.argv[2]
num_layers = int(sys.argv[3])

model = onnx.load(onnx_model, load_external_data=False)
model = onnx.shape_inference.infer_shapes(model)
load_external_data_for_model(
    model, os.path.dirname(onnx_model))
# Then the onnx_model has loaded the external data from the specific directory
# Then the onnx_model has converted raw data as external data and saved to specific directory
model_input_names = [i.name for i in model.graph.input]
model_output_names = [i.name for i in model.graph.output]
model_weights_names = [i.name for i in model.graph.initializer]
export_input_names = []
export_output_names = []
for node in model.graph.node[0:num_layers]:
    for i in node.input:
        if (i in model_input_names) or (i in model_weights_names):
            if i not in export_input_names:
                export_input_names.append(i)
        if i in export_output_names:
            export_output_names.remove(i)
    for o in node.output:
        export_output_names.append(o)

print(export_input_names)
print(export_output_names)

extract_model(onnx_model, subgraphx_path,
              export_input_names, export_output_names)
