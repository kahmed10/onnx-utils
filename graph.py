import onnx
import sys


if len(sys.argv) != 2:
    print("Usage: python graph.py [onnx_name]")
    exit()

onnx_name = sys.argv[1]

model = onnx.load(onnx_name)
print(onnx.helper.printable_graph(model.graph))
