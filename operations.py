import cv2
import numpy as np


def replace_color_with(image, center, radius, color):
    distance = find_color_distance(image, center)
    color_replaced_image = np.full_like(image, color)
    return np.where(distance <= radius, color_replaced_image, image)


def find_color_distance(image, color):
    color_difference = image - color
    distance = color_difference ** 2
    distance = distance.sum(axis=2)
    distance = distance ** 0.5
    distance = np.repeat(distance, 3)
    distance = distance.reshape(image.shape)
    return distance


def get_longest_line(lines):
    maximum_length = -1
    longest_line = lines[0]
    for line in lines:
        for x1, y1, x2, y2 in line:
            length = (((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** 0.5
            if length >= maximum_length:
                maximum_length = length
                longest_line = line
    return longest_line


def concatenate(array_1, array_2):
    if array_1 is None:
        return array_2
    if array_2 is None:
        return array_1
    return np.concatenate([array_1, array_2])


def get_largest_circle(circles):
    largest_radius = -1
    largest_circle = None
    for circle in circles:
        for x, y, radius in circle:
            if radius >= largest_radius:
                largest_radius = radius
                largest_circle = [[x, y, radius]]
    return largest_circle


def fill_hole(image, known_pixel):
    height, width = image.shape
    dimensions = (height, width)
    kernel = [(0, 255, 0), (255, 255, 255), (0, 255, 0)]
    kernel = np.array(kernel, 'uint8')
    compliment = image ^ 0xFF
    array = np.zeros(dimensions, 'uint8')
    array[known_pixel] = 255
    while True:
        result = np.bitwise_and(cv2.dilate(array, kernel), compliment)
        if np.array_equal(result, array):
            break
        array = result
    return np.bitwise_or(array, image)
