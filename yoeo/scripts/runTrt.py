import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class YOEOInference:
    def __init__(self, trt_model_path):
        # Load model
        self.load_trt_model(trt_model_path)

        # Declare input/outputs shapes based on the model
        self.get_io_shapes()

        # Allocate memory
        self.allocate_memory()

        self.stream = cuda.Stream()

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
        self.target_shape = self.input_shape[-2:] # get the two last dimensions of input shape: (416,416)

    def allocate_memory(self):
        # Allocate cuda memory for input and outputs
        self.input_memory = cuda.mem_alloc(int(np.prod(self.input_shape)) * np.dtype(np.float32).itemsize)
        self.detections_memory = cuda.mem_alloc(int(np.prod(self.detections_shape)) * np.dtype(np.float32).itemsize)
        self.segmentations_memory = cuda.mem_alloc(int(np.prod(self.segmentations_shape)) * np.dtype(np.float32).itemsize)

        # Allocate cpu memory for inputs and outputs
        self.detections_output = np.empty(self.detections_shape, dtype=np.float32)
        self.segmentations_output = np.empty(self.segmentations_shape, dtype=np.float32)
        self.segmented_image = np.zeros(self.target_shape, dtype=np.uint8)

    def preprocess_image(self, image):
        input_image = cv2.resize(image, self.target_shape)
        input_image = np.transpose(input_image, (2, 0, 1))  # HWC to CHW
        input_image = input_image / 255.0  # Normalize to [0, 1]
        input_image = np.expand_dims(input_image.astype(np.float32), axis=0)  # Add batch dimension
        input_image = np.ascontiguousarray(input_image)  # Ensure the array is contiguous
        
        return input_image

    def run_inference(self, input_image):           
        # Transfer input data to the GPU
        cuda.memcpy_htod_async(self.input_memory, input_image, self.stream)

        # Run inference
        bindings = [int(self.input_memory), int(self.detections_memory), int(self.segmentations_memory)]
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        # Transfer output data back to the host
        cuda.memcpy_dtoh_async(self.detections_output, self.detections_memory, self.stream)
        cuda.memcpy_dtoh_async(self.segmentations_output, self.segmentations_memory, self.stream)

        # Synchronize the stream
        self.stream.synchronize()
        return self.detections_output, self.segmentations_output

    def postprocess_segmentations(self, segmentations_output):
        segmentations_array = np.array(segmentations_output).squeeze()  # Remove batch dimension
        self.segmented_image[segmentations_array == 1] = 255  # Field markings
        self.segmented_image[segmentations_array == 2] = 127  # Field carpet

        return self.segmented_image

    def __call__(self, img):
        # Load and preprocess image
        input_image = self.preprocess_image(img)

        # Run inference
        _, self.segmentations_output = self.run_inference(input_image)

        # Postprocess and visualize results
        self.segmented_image = self.postprocess_segmentations(self.segmentations_output)
        
        return self.segmented_image


if __name__ == "__main__":
    import time

    trt_model_path = f'/home/drones/Documents/rc-fork/YOEO/config/model_fp16.trt'
    image_path = f'/home/drones/Documents/rc-fork/YOEO/data/samples/frame0256.jpg'

    # Run YOEO Inference
    yoeo = YOEOInference(trt_model_path)
    
    img = cv2.imread(image_path)

    for i in range(10):
        t0 = time.time()
        segmented_img = yoeo(img)
        print(f'elapsed time: {time.time() - t0:.3f}')

    cv2.imwrite('test.jpg', segmented_img)
