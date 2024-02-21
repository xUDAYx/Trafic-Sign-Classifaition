import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Loading trained model
model = tf.keras.models.load_model("model.h5")  

# Label Overview
class_num = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', +
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

# Define a function to preprocess input image
def preprocess_image(image):
    # Resize the image to the required dimensions
    image = image.resize((30, 30))
    # Convert the PIL image to a NumPy array
    image_array = np.array(image)
    # Normalize pixel values to be between 0 and 1
    image_array = image_array / 255.0
    # Add batch dimension to the image
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Define a function to make predictions
def predict_class(image):
    # Preprocess the input image
    processed_image = preprocess_image(image)
    # Make predictions using the loaded model
    predictions = model.predict(processed_image)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)
    # Return the predicted class label and class number
    return f"{class_num[predicted_class_index]} (Class {predicted_class_index})"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_class,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs="text"
)

# Launch the Gradio interface
iface.launch()
