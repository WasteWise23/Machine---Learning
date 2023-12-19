from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

img_path = r".\botol.jpg"  # Use raw string to avoid escape characters

# Load and preprocess the image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize pixel values to between 0 and 1

# Display the image
plt.imshow(img)
plt.show()

model = load_model(os.path.join('model_WasteWise_baru.h5'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
class_labels = ['battery', 'cardboard', 'carton', 'glass', 'metal', 'organic', 'paper', 'plastic']

images = np.vstack([img_array])
prediction = model.predict(images, batch_size=10)
class_index = np.argmax(prediction, axis=1)
class_label_prediction = class_labels[class_index[0]]


print(prediction)
print(class_index)
print(class_label_prediction)
