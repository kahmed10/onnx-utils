# onnx-utils

Utility scripts for editing or modifying onnx models. The script edits and modifies an onnx model to extract a subgraph based on input/output node names and shapes.

usage: onnx_edit.py [-h] [--inputs INPUTS] [--outputs OUTPUTS] [--skipverify]
                    input output

positional arguments:

  input              input onnx model  
  output             output onnx model
  

optional arguments:

  -h, --help             show this help message and exit
  
  --inputs INPUTS        comma separated model input names appended with shapes,
                         e.g. --inputs nodename1[1,2,3],nodename2[1,2,3]
  
  --outputs OUTPUTS      comma separated model output names appended with shapes,
                         e.g. --outputs outnodename1[1,2,3],outnodename2[1,2,3]
  
  --constants CONSTANTS  comma separated model constant names appended with shapes,
                         e.g. --constants constantnodename1[1,2,3],constantnodename2[1,2,3]
 
 --skipverify            skip verification of model. Useful if shapes are not known

# onnx-summarize

usage: onnx_summarize.py [-h] input

Creates a summary of operators in the input onnx file and also dumps each *loop
body* as an onnx file and also creates summary for each such loop body. 

positional arguments:
  input       input onnx model
