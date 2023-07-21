# Dataset: https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification

import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def extract_features(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (100, 100)) 
    flattened_image = resized_image.flatten()
    return flattened_image

def load_data(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        features = extract_features(image_path)
        images.append(features)
        labels.append(label)
    return images, labels

train_cat_folder = "train/cats"  
train_dog_folder = "train/dogs" 
train_cat_images, train_cat_labels = load_data(train_cat_folder, 0)
train_dog_images, train_dog_labels = load_data(train_dog_folder, 1)

train_images = train_cat_images + train_dog_images
train_labels = train_cat_labels + train_dog_labels

train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_cat_folder = "test/cats" 
test_dog_folder = "test/dogs"
test_cat_images, test_cat_labels = load_data(test_cat_folder, 0)
test_dog_images, test_dog_labels = load_data(test_dog_folder, 1)

test_images = test_cat_images + test_dog_images
test_labels = test_cat_labels + test_dog_labels

test_images = np.array(test_images)
test_labels = np.array(test_labels)

k = 6           #Accuracy: 0.5857142857142857
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(train_images, train_labels)

predictions = knn_classifier.predict(test_images)

accuracy = accuracy_score(test_labels, predictions)
print("Accuracy:", accuracy)

