import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2
import random


FIGURE_PATH = os.path.join(os.getcwd(), "figures")
os.makedirs(FIGURE_PATH, exist_ok=True)


def load_dataset():
    # List to store image paths and corresponding labels
    dataset_path = "arm_images"
    image_data = []
    labels = []

    file_names = os.listdir(dataset_path)
    random.shuffle(file_names)

    # Iterate over the files in the dataset directory
    for file_name in file_names:
        # Split the file name to get the label
        label, _ = file_name.split('_')  # Find the label associated with the image by splitting the
        # image on "_"

        # Append the image path and label to the lists
        image_data.append(cv2.imread(os.path.join(dataset_path, file_name)))

        # Convert label to integer (0 or 1)
        label = int(label)
        labels.append(label)
    # Convert labels into DataFrame
    target = pd.DataFrame(labels, columns=['label'])

    target['contracted'] = (target['label'] == 0).astype(int)
    target['extended'] = (target['label'] == 1).astype(int)

    target.drop(columns=['label'], inplace=True)

    # Divide into train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        image_data, target, test_size=0.2, random_state=42, stratify=target['contracted']
    )
    x_train = np.array(x_train).transpose(0, 3, 1, 2) / 255.0  # No need to reshape here
    x_test = np.array(x_test).transpose(0, 3, 1, 2) / 255.0  # No need to reshape here
    y_train = y_train.values
    y_test = y_test.values

    return x_train, x_test, y_train, y_test


def compute_accuracy(predictions, targets):
    return (predictions == np.argmax(targets, axis=1)).mean()

