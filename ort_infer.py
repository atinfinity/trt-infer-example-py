import onnxruntime
import numpy as np

model = "model/model_bn.onnx"
session = onnxruntime.InferenceSession(model, providers=['CPUExecutionProvider'])

# generate input
np.random.seed(0)
input_npa = np.random.randn(1, 3, 32, 32).astype(np.float32)

# inference
output_npa = session.run(["output"], {"input": input_npa})

print(output_npa[0])
