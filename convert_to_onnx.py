import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tf2onnx
import onnx

def convert():
    print("Loading MobileNetV2...")
    model = MobileNetV2(weights="imagenet")
    
    print("Converting to ONNX...")
    # Define input signature (1, 224, 224, 3)
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    
    output_path = "api/mobilenetv2.onnx"
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    convert()
