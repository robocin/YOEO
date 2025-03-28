import cv2
import numpy as np
import onnxruntime as ort
import argparse

# Main function
def inference(image_path, onnx_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        #TODO: Check if its appropriate to raise an error here
        raise FileNotFoundError(f"Image not found at {image_path}") 

    # Preprocess the image (assuming the model expects a 3xHxW normalized input)
    image_resized = cv2.resize(image, (416, 416))  # Adjust size if needed
    image_transposed = np.transpose(image_resized, (2, 0, 1))  # HWC to CHW
    image_normalized = image_transposed / 255.0  # Normalize to [0, 1]
    image_input = np.expand_dims(image_normalized.astype(np.float32), axis=0)  # Add batch dimension

    # Load the ONNX model
    session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[1].name
    output = session.run([output_name], {input_name: image_input})

    output_array = np.array(output[0]).squeeze()  # Remove the batch dimension
    print("Inference Output Shape:", output_array.shape)

    # Generate result image
    result_image = np.zeros_like(output_array, dtype=np.uint8)
    result_image[output_array == 1] = 255  # Field markings
    result_image[output_array == 2] = 127  # Field carpet

    # Save the result
    cv2.imwrite("segmented_result_onnx.jpg", result_image)

# Argument parser for CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference test with an ONNX model.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("onnx_path", type=str, help="Path to the ONNX model")

    args = parser.parse_args()

    # Call the main function with the arguments
    inference(args.image_path, args.onnx_path)
