# # Step 1: Import the necessary libraries and modules

import cv2
import numpy as np
import os
import glob
import string
import difflib
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


# # Step 2: Define the path for the dataset and initialize the lists for the images and labels

# dataset_path = "dataset/"
# images = []
# labels = []


# # Step 3: Load the images from the dataset and resize them to a standard size

# for i, label in enumerate(sorted(os.listdir(dataset_path))):
#     for filename in glob.glob(os.path.join(dataset_path, label, "*.jpg")):
#         image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#         image = cv2.resize(image, (32, 32))
#         images.append(image)
#         labels.append(label)


# # Step 4: Convert the images and labels to arrays and normalize the pixel values

# images = np.array(images) / 255.0
# labels = np.array(labels)


# # Step 5: Split the dataset into training and testing sets

# split = int(0.8 * len(images))
# train_images, test_images = images[:split], images[split:]
# train_labels, test_labels = labels[:split], labels[split:]


# # Step 6: Define the CNN model and compile it

# model = Sequential([
#     Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 1)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation="relu"),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation="relu"),
#     Flatten(),
#     Dropout(0.5),
#     Dense(128, activation="relu"),
#     Dropout(0.5),
#     Dense(len(string.ascii_uppercase), activation="softmax")
# ])

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"], run_eagerly=True)


# # Step 7: Train the model on the training set

# model.fit(train_images.reshape(-1, 32, 32, 1), train_labels, epochs=10, batch_size=64)


# # Step 8: Evaluate the model on the testing set

# test_loss, test_accuracy = model.evaluate(test_images.reshape(-1, 32, 32, 1), test_labels, verbose=2)
# print("Test accuracy:", test_accuracy)


# # Step 9: Load the handwritten prescription image and preprocess it

# prescription_image = cv2.imread("prescription.jpg", cv2.IMREAD_GRAYSCALE)
# prescription_image = cv2.threshold(prescription_image, 127, 255, cv2.THRESH_BINARY_INV)[1]


# # Step 10: Split the prescription image into characters and recognize them using the trained model

# recognized_characters = []

# ## Define a function to split the prescription image into individual characters. We can use the OpenCV library to perform this task. In this example, we assume that the characters in the prescription are all separated by a small margin.
# def split_characters(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     characters = []
#     for contour in contours:
#         [x, y, w, h] = cv2.boundingRect(contour)

#         if w < 10 or h < 10:
#             continue

#         character = gray[y:y+h, x:x+w]
#         characters.append(character)

#     return characters


# # Define function to preprocess individual characters
# def preprocess_character(img):
#     img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#     img = img / 255.0
#     img = img.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
#     return img

# # Define image size constants
# IMG_WIDTH = 64
# IMG_HEIGHT = 64


# ## Split the preprocessed prescription image into individual characters using the split_characters function.
# preprocessed_image = prescription_image
# characters = split_characters(preprocessed_image)


# # Define image size constants
# IMG_WIDTH = 64
# IMG_HEIGHT = 64

# for character in characters:
#     # Preprocess the character image
#     character = preprocess_character(character)

#     # Resize the character image to match the input size of the CNN model
#     character = cv2.resize(character, (IMG_WIDTH, IMG_HEIGHT))

#     # Reshape the character image to match the input shape of the CNN model
#     character = np.reshape(character, (1, IMG_WIDTH, IMG_HEIGHT, 1))

#     # Pass the character image through the trained CNN model to get the predicted label
#     predicted_label = model.predict(character)

#     # Append the predicted label to the list of recognized characters
#     recognized_characters.append(int2label[np.argmax(predicted_label)])



# recognized_string = ''.join(recognized_characters)



# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# import pytesseract

# # Step 1: Load the image of the prescription
# img = cv2.imread('dataset/crocin.png')

# # Step 2: Preprocess the image
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.bitwise_not(gray)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# # Step 3: Perform OCR on the preprocessed image to recognize the prescription text
# config = ("-l eng --oem 1 --psm 6")
# text = pytesseract.image_to_string(thresh, config=config)

# # Step 4: Clean and extract the relevant text from the recognized text
# text = text.replace('\n', ' ')
# text = ' '.join(text.split())
# text = text.upper()

# # Step 5: Match the extracted text with the medicine name database
# # matched_text = string_matching_algorithm(text)

# # Step 6: Identify the name of the doctor from the matched text
# # doctor_name = extract_doctor_name(matched_text)

# # Step 7: Load the training and testing data for character recognition
# (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
# train_images = train_images / 255.0
# test_images = test_images / 255.0

# # Step 8: Define the CNN model architecture
# model = keras.Sequential([
#     keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.Flatten(),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(10)
# ])
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# # Step 9: Train the model on the training set
# model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=5)



# # Define image size constants
# IMG_WIDTH = 64
# IMG_HEIGHT = 64

# # Define function to preprocess individual characters
# def preprocess_character(img):
#     img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#     img = img / 255.0
#     img = img.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
#     return img

# # Step 10: Split the prescription image into characters and recognize them using the trained model
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.bitwise_not(gray)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# characters = []
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     if w >= 5 and h >= 20:
#         roi = thresh[y:y+h, x:x+w]
#         resized_roi = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
#         preprocessed_roi = preprocess_character(resized_roi)
#         prediction = model.predict(preprocessed_roi.reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
#         characters.append(chr(np.argmax(prediction) + 48))

# recognized_text = ''.join(characters)

# # Step 11: Print the recognized text
# print(recognized_text)


import cv2

img = cv2.imread('dataset/cropped-prescription1.jpg')
# img = cv2.resize(img, [300,300])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("./output/output1.png", gray)

gray2 = cv2.bitwise_not(gray)
cv2.imwrite("./output/output2.png", gray)
thresh = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imwrite("output/output3.png", thresh)
thresh = cv2.GaussianBlur(thresh, (3,3), 0)
cv2.imwrite("output/output4.png", thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(contours)
print(hierarchy)

# cv2.imshow("a", thresh)
img_contour = cv2.drawContours(thresh, contours, -1, (0, 255, 0), 0)
cv2.imshow('Contours', img_contour)
cv2.imshow('Contours thresh', thresh)
cv2.waitKey(0)


# Define image size constants
IMG_WIDTH = 64
IMG_HEIGHT = 64
i=0

for cnt in contours:
    i=i+1
    x, y, w, h = cv2.boundingRect(cnt)
    if w >= 5 and h >= 20:
        roi = thresh[y:y+h, x:x+w]
        resized_roi = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(f'./output/contour{i}.png', resized_roi)
        cv2.imshow(f'contour{i}.png', resized_roi)
cv2.waitKey(0)


cv2.destroyAllWindows()