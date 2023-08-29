import tensorrt as trt
from cuda import cudart
import numpy as np


# deserialize TensorRT Engine
logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)
trt_engine_file = 'model/model_bn.onnx.engine'
with open(trt_engine_file, 'rb') as f:
    engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)

# create context
context = engine.create_execution_context()

# generate input
np.random.seed(0)
input_npa = np.random.randn(1, 3, 32, 32).astype(np.float32)
output_npa = np.zeros((1, 10), dtype=np.float32)

# allocate device memory
_, stream = cudart.cudaStreamCreate()
_, d_input_npa_ptr = cudart.cudaMallocAsync(input_npa.nbytes, stream)
_, d_output_npa_ptr = cudart.cudaMallocAsync(output_npa.nbytes, stream)

# copy HtoD
cudart.cudaMemcpyAsync(d_input_npa_ptr, input_npa.data, input_npa.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

# inference
bindings = [d_input_npa_ptr, d_output_npa_ptr]
context.execute_async_v2(bindings, stream)

# copy DtoH
cudart.cudaMemcpyAsync(output_npa.data, d_output_npa_ptr, output_npa.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
cudart.cudaStreamSynchronize(stream)

print(output_npa)

cudart.cudaFree(d_input_npa_ptr)
cudart.cudaFree(d_output_npa_ptr)
cudart.cudaStreamDestroy(stream)
