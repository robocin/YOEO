import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import os
from glob import glob

# Paths
image_dir = "/home/drones/Documents/YOEO/data/samples"
trt_model_path = "/home/drones/Documents/YOEO/config/model2.trt"
outputs_path = "/home/drones/Documents/YOEO/data/outputs"

# Get all .jpg and .png files in the directory
image_paths = glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png"))
if not image_paths:
    raise FileNotFoundError(f"No .jpg or .png images found in {image_dir}")

# Load TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(trt_model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# Allocate memory for inputs and outputs
context = engine.create_execution_context()

# Get input and output shapes
input_shape = engine.get_tensor_shape("InputLayer")
detections_shape = engine.get_tensor_shape("Detections")
segmentations_shape = engine.get_tensor_shape("Segmentations")

# Allocate device memory for input and outputs
input_memory = cuda.mem_alloc(int(np.prod(input_shape)) * np.dtype(np.float32).itemsize)
detections_memory = cuda.mem_alloc(int(np.prod(detections_shape)) * np.dtype(np.float32).itemsize)
segmentations_memory = cuda.mem_alloc(int(np.prod(segmentations_shape)) * np.dtype(np.float32).itemsize)

# Create a stream to perform inference asynchronously
stream = cuda.Stream()

# Initialize lists to store processing times
preprocess_times = []
inference_times = []
postprocess_times = []
total_times = []

run = 0

while run<20:
    # Process each image
    for image_path in image_paths:
        #print(f"Processing image: {image_path}")

        # Start total time measurement

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}. Skipping.")
            continue

        total_start_time = time.time()
        # Preprocess image
        preprocess_start_time = time.time()
        image_resized = cv2.resize(image, (416, 416))  # Adjust size if needed
        image_transposed = np.transpose(image_resized, (2, 0, 1))  # HWC to CHW
        image_normalized = image_transposed / 255.0  # Normalize to [0, 1]
        image_input = np.expand_dims(image_normalized.astype(np.float32), axis=0)  # Add batch dimension
        image_input = np.ascontiguousarray(image_input)  # Ensure the array is contiguous
        preprocess_end_time = time.time()
        preprocess_time = (preprocess_end_time - preprocess_start_time) * 1000

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
        inference_time = (inference_end_time - inference_start_time) * 1000

        # Postprocess and visualize results
        postprocess_start_time = time.time()
        segmentations_array = np.array(segmentations_output).squeeze()  # Remove batch dimension if needed

        # Create a blank image for visualization
        segmented_image = np.zeros_like(segmentations_array, dtype=np.uint8)
        segmented_image[segmentations_array == 1] = 255  # Field markings
        segmented_image[segmentations_array == 2] = 127  # Field carpet

        # Save the segmented image
        output_path = os.path.join(outputs_path, f"segmented_{os.path.basename(image_path)}")
        postprocess_end_time = time.time()
        total_end_time = time.time()
        postprocess_time = (postprocess_end_time - postprocess_start_time) * 1000
        cv2.imwrite(output_path, segmented_image)

        # End total time measurement
        total_time = (total_end_time - total_start_time) * 1000

        if run!=0:
            preprocess_times.append(preprocess_time)
            inference_times.append(inference_time)
            postprocess_times.append(postprocess_time)
            total_times.append(total_time)
            print(f"Processed {image_path}'s inference in {inference_time:.2f} ms")
    
    run+=1

# Compute average processing times
avg_preprocess_time = np.mean(preprocess_times)
avg_inference_time = np.mean(inference_times)
avg_postprocess_time = np.mean(postprocess_times)
avg_total_time = np.mean(total_times)

# Print results
print("\nAverage Processing Times:")
print(f"Preprocessing: {avg_preprocess_time:.2f} ms")
print(f"Inference: {avg_inference_time:.2f} ms")
print(f"Postprocessing: {avg_postprocess_time:.2f} ms")
print(f"Total: {avg_total_time:.2f} ms")
print()