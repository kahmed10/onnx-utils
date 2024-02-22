import onnx
import sys

onnx_name = sys.argv[1]

if len(sys.argv) != 2:
    print("Usage: python graph.py [onnx_name]")
    exit()

model = onnx.load(onnx_name)
print(onnx.helper.printable_graph(model.graph))
