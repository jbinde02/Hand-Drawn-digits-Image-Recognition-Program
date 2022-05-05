import os
import pickle
import tkinter

import numpy
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn import svm

# https://howtocreateapps.com/image-recognition-python/
# https://www.techwithtim.net/tutorials/python-module-walk-throughs/turtle-module/drawing-with-mouse/


def create_trained_model(model_filename):
    # This bit trains the model to predict digits
    # Loads the data
    digits_data = load_digits()
    # Get the total number of samples
    img_samples = len(digits_data.images)
    # Get the handwritten images
    img = digits_data.images.reshape(img_samples, -1)
    # Get the target labels
    labels = digits_data.target
    # The model
    classify = svm.SVC(gamma=0.001)
    # flatten sample images are stored in img variable
    img_half = img[:img_samples // 2]
    # target labels are stored in labels variable
    labels_half = labels[:img_samples // 2]
    # Training: First is half the images, second is all the images
    # classify.fit(img_half, labels_half)
    classify.fit(img, labels)
    # End training
    # Save model
    with open(model_filename, 'wb') as file:
        pickle.dump(classify, file)


def load_model(model_filename):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model


def show_image(image):
    plt.matshow(image)
    plt.show()


# Predicts half of the images dataset
def example():
    # This bit trains the model to predict digits
    # Loads the data
    digits_data = load_digits()
    # Get the total number of samples
    img_samples = len(digits_data.images)
    # Get the handwritten images
    img = digits_data.images.reshape(img_samples, -1)
    # Get the target labels
    labels = digits_data.target
    # The model
    classify = svm.SVC(gamma=0.001)
    # flatten sample images are stored in img variable
    img_half = img[:img_samples // 2]
    # target labels are stored in labels variable
    labels_half = labels[:img_samples // 2]
    # Training: First is half the images, second is all the images
    # classify.fit(img_half, labels_half)
    classify.fit(img, labels)
    # End training

    images = list(zip(digits_data.images, digits_data.target))
    plt.figure(figsize=(4, 4))
    for i, (image, label) in enumerate(images[:10]):
        # initializing subplot of 3x5
        plt.subplot(3, 5, i + 1)
        # display images in the subplots
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        # set title for each subplot
        plt.title("Training: %i" % label)

    # display the plot
    plt.show()

    # Predict half the images
    labels_expected = digits_data.target[img_samples // 2:]
    img_predicted = classify.predict(img[img_samples // 2:])

    images_predictions = list(zip(digits_data.images[img_samples // 2:], img_predicted))

    for i, (image, predict) in enumerate(images_predictions[:10]):
        # initialize the subplot of size 3x5
        plt.subplot(3, 5, i + 1)
        # turn of the axis values (the labels for each value in x and y axis)
        plt.axis('off')
        # display the predicted images in the subplot
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        # set the title for each subplot in the main plot
        plt.title("Predict: %i" % predict)

    plt.show()

    print("Classification Report %s:\n%s\n"
          % (classify, metrics.classification_report(labels_expected, img_predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels_expected, img_predicted))


# Give list of images that are made up of 64 values each
def predict_images(model, images, labels_expected=None):
    # Format parameters
    for i in range(len(images)):
        # Turn input image into numpy.ndarray
        if not isinstance(images[i], numpy.ndarray):
            try:
                print("Converting input image of type" + str(type(images[i])) + " to type numpy.ndarray")
                images[i] = numpy.array(images[i])
            except Exception as e:
                print(e)
                print("Failure to convert image input to type numpy.ndarray")
        numpy.array([labels_expected])

    # Predict with trained model
    imgs_predicted = model.predict(images)
    print("This image is predicted to be a " + str(imgs_predicted))

    # Plot the results
    plt.figure(num="Prediction Results", figsize=(15, 7))  # Edits the window title and size
    for i, image in enumerate(images):
        reshaped_image = image.reshape(8, 8)
        plt.subplot(3, 10, i+1)  # Only works with up to 30 digits atm
        plt.axis("off")
        plt.imshow(reshaped_image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title("Expected: %i \n Predicted: %i" % (labels_expected[i], imgs_predicted[i]))
    plt.show()

    if labels_expected is not None:
        print("Classification Report %s:\n%s\n"
              % (model, metrics.classification_report(labels_expected, imgs_predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels_expected, imgs_predicted))


def process_image(filename):
    img = Image.open(filename)  # 64x64 size drawing of digit
    img = img.resize((8, 8))  # resize to 8x8
    pixels = list(img.getdata())
    formatted_pixels = []
    min_pixel = 256
    max_pixel = 0
    for pixel in pixels:
        fpixel = abs(pixel[0] - 255)  # This is to make it so that white will end up with a value of zero and shades of black greater than 0
        formatted_pixels.append(fpixel)
        if fpixel > max_pixel:
            max_pixel = fpixel
        if fpixel < min_pixel:
            min_pixel = fpixel

    normalized_pixels = list(map(lambda pix: round((pix - min_pixel) / (max_pixel - min_pixel) * 15), formatted_pixels))
    return normalized_pixels


def main(train_model=True):
    model_filename = "handdrawn_digits_model.pkl"
    if train_model:
        print("Training new hand drawn digits model.")
        create_trained_model(model_filename)
    model = load_model(model_filename)
    # Get hand drawn number files and process each, and putting them in a list
    handdrawn_numbers_path = os.path.dirname(os.path.realpath(__file__))
    images = []
    for (root, dirs, file) in os.walk(handdrawn_numbers_path):
        for f in file:
            if '.png' in f:
                images.append(process_image(os.path.join(root, f)))

    # Predict drawn digit
    # List of labels that show what each digit actually is
    actual_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    predict_images(model, images, actual_values)


# Start here
train_model = False
main(train_model)

# Precision = True Positive/(True Positive + False Positive)
# Recall = True Positive/(True Positive + False Negative)
# F1-Score = 2/((1/Recall) + (1/Precision))
# Confusion Matrix: Columns = A labels Precision, Rows = A labels Recall

# May be useful
# https://stackoverflow.com/questions/41666627/using-own-image-in-sklearn-digits-example
# https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
