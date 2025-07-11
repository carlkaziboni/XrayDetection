import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import os

saved_model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'xray_detection_transfer_model.keras'))
xception_model = saved_model.get_layer('xception')

prediction_labels = ['Consolidation',
 'Atelectasis',
 'Nodule',
 'Hernia',
 'Edema',
 'Effusion',
 'Cardiomegaly',
 'Mass',
 'Emphysema',
 'Fibrosis',
 'Pleural_Thickening',
 'No Finding',
 'Pneumonia',
 'Infiltration',
 'Pneumothorax']

def get_predictions(image_path):
    """
    Predict findings for a given X-ray image.
    
    Args:
        image_path (str): Path to the X-ray image.
    
    Returns:
        list: List of predicted findings.
    """
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)
    
    predictions = saved_model.predict(img_array)
    predicted_labels = [prediction_labels[i] for i in range(len(predictions[0])) if predictions[0][i] > 0.4]
    
    return predicted_labels

# Grad CAM visualization for transfer model
def get_grad_cam_heatmap_transfer(img_array,model=xception_model,layer_name='block14_sepconv2_act', pred_index=None):
    """
    Generate Grad CAM heatmap for a given image and transfer model layer.
    
    Args:
        model (tf.keras.Model): Trained transfer model.
        img_array (np.ndarray): Preprocessed image array.
        layer_name (str): Name of the convolutional layer to visualize.
    
    Returns:
        np.ndarray: Heatmap of the Grad CAM.
    """
    # Ensure input is a TensorFlow tensor
    if isinstance(img_array, np.ndarray):
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    assert isinstance(img_array, tf.Tensor), "img_array must be a TensorFlow tensor"
    assert len(img_array.shape) == 4, "img_array must have shape (1, height, width, channels)"

    # Create a model that maps the input to the activations and predictions
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pass
            #pred_index = tf.argmax(predictions[0]).numpy().item()
        loss = tf.reduce_max(predictions)

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()