from imgaug import augmenters as iaa
import cv2
import numpy as np


def zoom(image):
    zoom_img = iaa.Affine(scale=(1, 1.3))
    image = zoom_img.augment_image(image)
    return image

# Function to flip the image and steering angle
def img_random_flip(image, choice):
    image = cv2.flip(image, 1)
    steering = choice[0]
    throttle = choice[1]
    steering = -steering
    new_choice = [steering, throttle]
    return image, new_choice


def pan(image):
    pan = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image


def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image


def random_augment(image, choice):
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, choice = img_random_flip(image, choice)

    return image, choice
