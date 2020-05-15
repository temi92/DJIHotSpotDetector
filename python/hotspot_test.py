# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:36:59 2020

@author: Drones
"""
import pickle
import tensorflow as tf
import numpy as np
import math
from scipy.special import expit



def preprocess_image(file_image, image_size):
    #convert to numpy array
    image = tf.keras.preprocessing.image.load_img(file_image, target_size=image_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.imagenet_utils.preprocess_input(image)
    return image


def run_interpreter(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data



model_mobileNet = tf.keras.applications.MobileNetV2(input_shape = (224 ,224,3), weights="imagenet", include_top=False)
with open("model.cpickle", "rb") as f:
    model = pickle.load(f)
    

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="mobileNetv2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



image = preprocess_image("living_room.jpg", (224, 224))   


#extact features using tf api...
preds = run_interpreter(image)
    

labels = ["fire", "no_fire"]
#extract features using MobileNetV2 ..

preds = model_mobileNet.predict(image)



#flatten array

preds = preds.reshape((preds.shape[0], 1280 * 7 * 7))


#use logistic regression to get prediction from features computed by MobileNet V2..
#if pred_label is 0, fire  present and if 1 no-fire is present.

#pred_label  = model.predict(preds)[0] 

#compute predictions..
predictions = model.predict_proba(preds)



#my implementation for predictions...
prob = np.matmul(preds, model.coef_.T) + model.intercept_

#pass through logistic sigmoid function..
prob = 1/(1+ np.exp(-prob)) 
print(prob)
#prob = expit(prob, out=prob)


prob = np.vstack([1-prob, prob]).T
index = prob.argmax(axis=1)
accuracy = prob[:, index].ravel()[0]


predicted_label = np.array(labels)[index][0]

print ("accuracy is {} and label is {}".format(accuracy, predicted_label))





