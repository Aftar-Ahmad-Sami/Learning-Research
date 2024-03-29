#Installing Libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from google.colab.patches import cv2_imshow
from google.colab import drive

# Loading Data from Google Drive
drive.mount('/content/drive')
path_of_test = '/content/drive/MyDrive/4TH YEAR - Thesis (Primary)/Datasets/1/breast-cancer-dataset/Test'
path_of_train = '/content/drive/MyDrive/4TH YEAR - Thesis (Primary)/Datasets/1/breast-cancer-dataset/Train'

"""# **Data Load with Preprocessing and Visualization**"""

def load_data_with_preprocessing(des_path,size):
  images = []
  labels = []
  for name in os.listdir(des_path):
    path = os.path.join(des_path,name) # have the full path
    _,filetype = os.path.splitext(path) #stores the extension
    if filetype == '.png':
      img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
      img = cv2.resize(img,(size,size))
      images.append(np.array(img))
      label = 0 if "Benign" in des_path else 1  # 0 for Benign, 1 for Malignant
      labels.append(label)
  return images, labels

benign_train, benign_train_labels = load_data_with_preprocessing(path_of_train + '/Benign', 244)
malign_train, malign_train_labels = load_data_with_preprocessing(path_of_train +  '/Malignant', 244)
benign_test, benign_test_labels = load_data_with_preprocessing(path_of_test +  '/Benign', 244)
malign_test, malign_test_labels = load_data_with_preprocessing(path_of_test +  '/Malignant', 244)


# VISUALIZATION
print(f"Total Benign train samples: {len(benign_train)}")
print(f"Total Malignant train samples: {len(malign_train)}")
print(f"Total Benign test samples: {len(benign_test)}")
print(f"Total Malignant test samples: {len(malign_test)}")

def visualization(images, labels, class_name):
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap='gray') # cmap for grayscale demo
        plt.title(class_name[labels[i]])
        plt.axis('on')
    plt.show()

visualization(benign_train, benign_train_labels, class_name=["Benign", "Malignant"])
visualization(malign_train, malign_train_labels, class_name=["Benign", "Malignant"])
visualization(benign_test, benign_test_labels, class_name=["Benign", "Malignant"])
visualization(malign_test, malign_test_labels, class_name=["Benign", "Malignant"])
