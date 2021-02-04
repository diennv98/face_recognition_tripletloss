import os
import pickle
import numpy as np
import glob
import cv2
from keras.models import load_model

class Model():
    def __init__(self, model_path, SIZE):
        self.model = load_model(model_path)
        self.SIZE = SIZE

    def save_pickle(self, obj, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    def load_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def normalize(self, X):
        axis = (0, 1, 2)
        mean = np.mean(X, axis)
        std = np.std(X, axis)
        size = X.size
        adj_std = np.maximum(std, 1 / np.sqrt(size))
        X = (X - mean) / adj_std
        return X

    def image_read(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_norm = self.normalize(image)
        return image_norm

    def distance(self, a, b):
        a /= np.sqrt(np.maximum(np.sum(np.square(a)), 1e-10))
        b /= np.sqrt(np.maximum(np.sum(np.square(b)), 1e-10))

        return np.sqrt(np.sum(np.square(a - b)))

    def get_embedding(self, image_path):
        X = self.image_read(image_path)
        X = np.expand_dims(X, axis=0)
        embedding = self.model.predict(X)
        return embedding

    def write_embedding(self, dataset, embeddings_path_save, labels_path_save):
        # save embedding and label from dataset
        # example embedding_path_save and labels_path_save: embeddings.pkl and labels.pkl
        labels = []
        embeddings = []
        folders = os.listdir(dataset)
        for fd in folders:
            files = glob.glob(dataset + '/' + fd + '/*')
            for f in files:
                embedding = self.get_embedding(f)
                labels.append(fd)
                embeddings.append(embedding)

        self.save_pickle(embeddings, embeddings_path_save)
        self.save_pickle(labels, labels_path_save)

    def find_face(self, image_path, embeddings_path, labels_path, dataset = None, write_embedding = False):
        # find min distance input with all embedding in dataset
        # example embedding_path_save and labels_path_save: embeddings.pkl and labels.pkl

        if write_embedding == True:
            self.write_embedding(dataset, embeddings_path, labels_path)

        embeddings = self.load_pickle(embeddings_path)
        labels = self.load_pickle(labels_path)

        embedding = self.get_embedding(image_path)
        min = 9999
        label = None
        for i in range(len(embeddings)):
            value = self.distance(embedding, embeddings[i])
            if value < min:
                min = value
                label = labels[i]

        return label
