import numpy as np
import cv2
from keras.models import load_model
from data_processing import normalize

def image_read(image_path, SIZE):
    image = cv2.imread(image_path)
    image = cv2.resize(image, SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_norm = normalize(image)
    return image_norm

def distance(a, b):
    a/= np.sqrt(np.maximum(np.sum(np.square(a)), 1e-10))
    b/= np.sqrt(np.maximum(np.sum(np.square(b)), 1e-10))

    return np.sqrt(np.sum(np.square(a-b)))

SIZE = (160, 160)
image_path = ''

# load model
model = load_model('model.h5')

# predict and return embedding is vector 128d
image = image_read(image_path, SIZE)
X = np.expand_dims(image, axis = 0)
embedding = model.predict(X)
