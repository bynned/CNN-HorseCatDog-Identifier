# -*- coding: utf-8 -*-
"""
Created on Mon Dec 4 17:56:12 2023

@author: Benny
"""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Load the model
model = load_model('30epoch_model')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    './test_set',
    target_size=(300, 300),
    batch_size=1,
    class_mode='categorical',
    shuffle=False  # do not shuffle to maintain the order
)


test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict the test set
test_predictions = model.predict(test_generator)
predicted_classes = np.argmax(test_predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)

# classification report
class_report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print("Classification Report:")
print(class_report)

