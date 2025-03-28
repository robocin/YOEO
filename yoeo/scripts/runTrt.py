import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class YOEOInference:
    def __init__(self, trt_model_path, target_size=(416, 416)):
        self.target_size = target_size
        self.engine = None
        self.context = None
        self.input_memory = None
        self.detections_memory = None
        self.segmentations_memory = None
        self.stream = None

        ## Init routine
        # Load model
        self.load_trt_model(trt_model_path)
        # allocar memoria de acordo com o target_size
        self.get_io_shapes() #Values of image_input, detections_shape, segmentations_shape are updated to the class
        self.allocate_memory()


    def preprocess_image(self, image):
        image_resized = cv2.resize(image, self.target_size)
        image_transposed = np.transpose(image_resized, (2, 0, 1))  # HWC to CHW
        image_normalized = image_transposed / 255.0  # Normalize to [0, 1]
        image_input = np.expand_dims(image_normalized.astype(np.float32), axis=0)  # Add batch dimension
        image_input = np.ascontiguousarray(image_input)  # Ensure the array is contiguous
        return image_input

    def load_trt_model(self, trt_model_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(trt_model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        # Run after loading the model
        self.context = self.engine.create_execution_context()

    def get_io_shapes(self):
        # Get input and output shapes
        self.input_shape = self.engine.get_tensor_shape("InputLayer")
        self.detections_shape = self.engine.get_tensor_shape("Detections")
        self.segmentations_shape = self.engine.get_tensor_shape("Segmentations")

        #return self.input_shape, self.detections_shape, self.segmentations_shape

    def allocate_memory(self):
        # Allocate device memory for input and outputs
        self.input_memory = cuda.mem_alloc(int(np.prod(self.input_shape)) * np.dtype(np.float32).itemsize)
        self.detections_memory = cuda.mem_alloc(int(np.prod(self.detections_shape)) * np.dtype(np.float32).itemsize)
        self.segmentations_memory = cuda.mem_alloc(int(np.prod(self.segmentations_shape)) * np.dtype(np.float32).itemsize)
        self.stream = cuda.Stream()

    def run_inference(self, image_input):           
        # Transfer input data to the GPU
        cuda.memcpy_htod_async(self.input_memory, image_input, self.stream)

        # Run inference
        bindings = [int(self.input_memory), int(self.detections_memory), int(self.segmentations_memory)]
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        # Transfer output data back to the host
        detections_output = np.empty(self.detections_shape, dtype=np.float32)
        segmentations_output = np.empty(self.segmentations_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(detections_output, self.detections_memory, self.stream)
        cuda.memcpy_dtoh_async(segmentations_output, self.segmentations_memory, self.stream)

        # Synchronize the stream
        self.stream.synchronize()
        return detections_output, segmentations_output

    def postprocess_segmentations(self, segmentations_output):
        segmentations_array = np.array(segmentations_output).squeeze()  # Remove batch dimension if needed

        # Create a blank image for visualization
        segmented_image = np.zeros_like(segmentations_array, dtype=np.uint8)
        segmented_image[segmentations_array == 1] = 255  # Field markings
        segmented_image[segmentations_array == 2] = 127  # Field carpet

        #return resultado

        return segmented_image

    def __call__(self, img):
        """ 
        Run YOEO Inference with TensorRT model 
        Args:
            img (np.ndarray): Input image to run inference on
        Returns:
            np.ndarray: Segmented image
        """
        # Load and preprocess image
        image_input = self.preprocess_image(img)

        
    

        # Run inference
        detections_output, segmentations_output = self.run_inference(image_input)

        # Postprocess and visualize results
        segmented_image = self.postprocess_segmentations(segmentations_output)
        
        cv2.imwrite("segmented_result_trt.jpg", segmented_image)
        return segmented_image


if __name__ == "__main__":
    import argparse
    
    # Setup argparse to handle command line arguments
    parser = argparse.ArgumentParser(description="Run YOEO Inference with TensorRT model")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("trt_model_path", type=str, help="Path to the TensorRT model")

    # Parse the arguments
    args = parser.parse_args()

    # Run YOEO Inference
    yoeo = YOEOInference(args.trt_model_path)
    img = cv2.imread(args.image_path)
    yoeo(img)

