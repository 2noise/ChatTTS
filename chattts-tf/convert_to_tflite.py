import tensorflow as tf
import os


def convert_to_tflite():
    # Load the SavedModel
    model = tf.saved_model.load("chattts_tf")

    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_saved_model("chattts_tf")

    # Set optimization flags
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]

    # Convert the model
    tflite_model = converter.convert()

    # Save the TFLite model
    with open("chattts_model.tflite", "wb") as f:
        f.write(tflite_model)

    print("Model converted successfully to TFLite format")


if __name__ == "__main__":
    convert_to_tflite()
