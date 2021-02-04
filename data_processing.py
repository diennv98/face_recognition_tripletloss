import os
import numpy as np
import cv2

def normalize(X):
    axis = (0,1,2)
    mean = np.mean(X, axis)
    std = np.std(X, axis)
    size= X.size
    adj_std = np.maximum(std, 1/np.sqrt(size))
    X = (X-mean)/adj_std
    return X

# load data and label from dataset
def load_data_label(dataset_path):
    SIZE = (160, 160)

    faces = []
    face_labels = []

    for folder in os.listdir(dataset_path):
      for filename in os.listdir(os.path.join(dataset_path, folder)):
        face = cv2.imread(dataset_path + '/' + folder + '/' + filename)
        face = cv2.resize(face, SIZE)
        faces.append(normalize(face))
        face_labels.append(folder)

    face_labels = np.array(face_labels)
    faces = np.array(faces)
    return faces, face_labels

# get three input anchor, positive and negative
def get_data(dataset_path):
    faces, face_labels = load_data_label(dataset_path)

    X_anchor = []
    X_positive = []
    X_anchor_labels = []

    label_list = np.unique(face_labels)

    for label in label_list:
      filter = (face_labels == label).reshape(faces.shape[0])
      X_face = faces[filter]

      for i1 in range(X_face.shape[0]):
        for i2 in range(i1 + 1, X_face.shape[0]):
          X_anchor.append(X_face[i1])
          X_positive.append(X_face[i2])
          X_anchor_labels.append(label)

    X_anchor = np.array(X_anchor)
    X_positive = np.array(X_positive)
    X_anchor_labels = np.array(X_anchor_labels)

    X_negative = []
    for i in range(X_anchor.shape[0]):
        flag = False
        while (not flag):
            index = np.random.randint(0, faces.shape[0])
            if face_labels[index] != X_anchor_labels[i]:
                X_negative.append(faces[index])
                flag = True

    X_negative = np.array(X_negative)

    print('Anchor: ', X_anchor.shape)
    print('Positive: ', X_positive.shape)
    print('Negative: ', X_negative.shape)
    return [X_anchor, X_positive, X_negative]