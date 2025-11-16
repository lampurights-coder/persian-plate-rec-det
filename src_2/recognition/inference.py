import onnxruntime as ort
import os
import sys
import logging

logger = logging.getLogger('ppocr')

class ONNXInferenceSession:
    def __init__(self, model_path, use_gpu=False, gpu_id=0):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.session = self._create_session()
        self.input_names = self._get_input_names()
        
    def _create_session(self):
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model file not found: {self.model_path}")
        
        sess_options = ort.SessionOptions()
        
        if self.use_gpu:
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": self.gpu_id,
                    "cudnn_conv_algo_search": "DEFAULT"
                })
            ]
        else:
            providers = ["CPUExecutionProvider"]
            
        session = ort.InferenceSession(
            self.model_path,
            providers=providers,
            sess_options=sess_options,
        )
        return session
    
    def _get_input_names(self):
        inputs = self.session.get_inputs()
        if len(inputs) == 1:
            return inputs[0].name
        else:
            return [input.name for input in inputs]
    
    def run(self, input_data):
        if isinstance(self.input_names, list):
            input_dict = {}
            if isinstance(input_data, dict):
                input_dict = input_data
            else:
                # Handle multiple inputs
                for i, name in enumerate(self.input_names):
                    input_dict[name] = input_data[i]
        else:
            input_dict = {self.input_names: input_data}
            
        outputs = self.session.run(None, input_dict)
        return outputs