import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time  # For measuring time

# Paths
image_path = "/home/drones/Documents/YOEO/data/samples/frame6.png"
trt_model_path = "/home/drones/Documents/YOEO/config/model.trt"

# Start total time measurement
total_start_time = time.time()

# Load image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Preprocess image (assuming model expects 3xHxW normalized input)
preprocess_start_time = time.time()
image_resized = cv2.resize(image, (416, 416))  # Adjust size if needed
image_transposed = np.transpose(image_resized, (2, 0, 1))  # HWC to CHW
image_normalized = image_transposed / 255.0  # Normalize to [0, 1]
image_input = np.expand_dims(image_normalized.astype(np.float32), axis=0)  # Add batch dimension
image_input = np.ascontiguousarray(image_input)  # Ensure the array is contiguous
preprocess_end_time = time.time()
print(f"Preprocessing time: {(preprocess_end_time - preprocess_start_time) * 1000:.2f} ms")

# Load TensorRT engine
load_engine_start_time = time.time()
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(trt_model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
load_engine_end_time = time.time()
print(f"Engine loading time: {(load_engine_end_time - load_engine_start_time) * 1000:.2f} ms")

# Allocate memory for inputs and outputs
context = engine.create_execution_context()

# Get input and output shapes
input_shape = engine.get_tensor_shape("InputLayer")
detections_shape = engine.get_tensor_shape("Detections")
segmentations_shape = engine.get_tensor_shape("Segmentations")

# Allocate device memory for input and outputs
input_memory = cuda.mem_alloc(image_input.nbytes)
detections_memory = cuda.mem_alloc(int(np.prod(detections_shape)) * np.dtype(np.float32).itemsize)
segmentations_memory = cuda.mem_alloc(int(np.prod(segmentations_shape)) * np.dtype(np.float32).itemsize)

# Create a stream to perform inference asynchronously
stream = cuda.Stream()

# Transfer input data to the GPU
cuda.memcpy_htod_async(input_memory, image_input, stream)

# Run inference
inference_start_time = time.time()
bindings = [int(input_memory), int(detections_memory), int(segmentations_memory)]
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

# Transfer output data back to the host
detections_output = np.empty(detections_shape, dtype=np.float32)
segmentations_output = np.empty(segmentations_shape, dtype=np.float32)
cuda.memcpy_dtoh_async(detections_output, detections_memory, stream)
cuda.memcpy_dtoh_async(segmentations_output, segmentations_memory, stream)

# Synchronize the stream
stream.synchronize()
inference_end_time = time.time()
print(f"Inference time: {(inference_end_time - inference_start_time) * 1000:.2f} ms")

# Postprocess and visualize results
postprocess_start_time = time.time()
# Detections output
print("Detections Output Shape:", detections_output.shape)
# Process detections_output as needed (e.g., bounding boxes, class labels, etc.)

# Segmentations output
print("Segmentations Output Shape:", segmentations_output.shape)
segmentations_array = np.array(segmentations_output).squeeze()  # Remove batch dimension if needed

# Create a blank image for visualization
segmented_image = np.zeros_like(segmentations_array, dtype=np.uint8)
segmented_image[segmentations_array == 1] = 255  # Field markings
segmented_image[segmentations_array == 2] = 127  # Field carpet

# Save the segmented image
cv2.imwrite("segmented_result_trt.jpg", segmented_image)
postprocess_end_time = time.time()
print(f"Postprocessing time: {(postprocess_end_time - postprocess_start_time) * 1000:.2f} ms")

# End total time measurement
total_end_time = time.time()
print(f"Total processing time: {(total_end_time - total_start_time) * 1000:.2f} ms")