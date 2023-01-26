import streamlit as st
import tensorflow as tf
import os
import numpy as np
class_names = ['ceket', 'esofman', 'gomlek', 'kazak', 'mont', 'pantolon', 'sort', 'takim', 'tisort', 'yelek']
 
## Page Title
st.set_page_config(page_title = "Trendyol Image Classification")
st.title(" Trendyol Image Classification")
st.markdown("---")

 
tflite_interpreter = tf.lite.Interpreter(model_path='converted_model.tflite')
tflite_interpreter.allocate_tensors()
 
def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image
 
def get_predictions(input_image):
    output_details = tflite_interpreter.get_output_details()
    set_input_tensor(tflite_interpreter, input_image)
    tflite_interpreter.invoke()
    tflite_model_prediction = tflite_interpreter.get_tensor(output_details[0]["index"])
    tflite_model_prediction = tflite_model_prediction.squeeze().argmax(axis = 0)
    pred_class = class_names[tflite_model_prediction]
    return pred_class
 
 
## Input Fields
uploaded_file = st.file_uploader("Upload a Image", type=["jpg","png", 'jpeg'])
if uploaded_file is not None:
    with open(os.path.join("/tmp",uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer())
    path = os.path.join("/tmp",uploaded_file.name)
    img = tf.keras.preprocessing.image.load_img(path , grayscale=False, color_mode='rgb', target_size=(224,224,3), interpolation='nearest')
    st.image(img)
    #print(value)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
 
 
if st.button("Get Predictions"):
    suggestion = get_predictions(input_image =img_array)
    st.success(suggestion)