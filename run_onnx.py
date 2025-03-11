import cv2
import numpy as np
import onnxruntime as ort

# Paths
image_path = "/home/drones/Documents/YOEO/data/samples/frame6.png"
onnx_model_path = "/home/drones/Documents/YOEO/config/model_fixed.onnx"

# Load image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Preprocess image (assuming model expects 3xHxW normalized input)
image_resized = cv2.resize(image, (416, 416))  # Adjust size if needed
image_transposed = np.transpose(image_resized, (2, 0, 1))  # HWC to CHW
image_normalized = image_transposed / 255.0  # Normalize to [0, 1]
image_input = np.expand_dims(image_normalized.astype(np.float32), axis=0)  # Add batch dimension

# Load ONNX model
session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[1].name
output = session.run([output_name], {input_name: image_input})

# Postprocess and visualize result (modify as needed)
output_array = np.array(output[0]).squeeze()  # Remove batch dimension
print("Inference Output Shape:", output_array.shape)

image = np.zeros_like(output_array, dtype=np.uint8)
image[output_array == 1] = 255  # Field markings
image[output_array == 2] = 127  # Field carpet

# Display result
cv2.imwrite("segmented_result_onnx.jpg", image)
