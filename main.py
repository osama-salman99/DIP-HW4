from operations import *


def find_objects(color_image, largest_circle_color, longest_line_color, output_folder_name):
    global final_image

    # Write color-split image to the output folder
    cv2.imwrite(f'output\\{output_folder_name}\\03_color_image.png', color_image)

    # Threshold the image to eliminate low-valued pixels
    _, thresholded_image = cv2.threshold(color_image, thresh=115, maxval=255, type=cv2.THRESH_BINARY)

    # Write thresholded image to the output folder
    cv2.imwrite(f'output\\{output_folder_name}\\04_thresholded_image.png', thresholded_image)

    # Find the lines in the image
    lines = cv2.HoughLinesP(thresholded_image, rho=1, theta=np.pi / 180, threshold=200, minLineLength=200,
                            maxLineGap=60)

    # Draw all found lines
    all_lines_image = np.zeros(image_dimensions, 'uint8')
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(all_lines_image, pt1=(x1, y1), pt2=(x2, y2), color=255, thickness=2)

    # Write all lines image to the output folder
    cv2.imwrite(f'output\\{output_folder_name}\\05_all_lines_image.png', all_lines_image)

    # Find the longest line
    longest_line = get_longest_line(lines)

    # Draw the longest line on the final image and on the longest line image
    longest_line_image = np.zeros(image_dimensions, 'uint8')
    for x1, y1, x2, y2 in longest_line:
        cv2.line(longest_line_image, (x1, y1), (x2, y2), 255, 2)
        cv2.line(final_image, pt1=(x1, y1), pt2=(x2, y2), color=longest_line_color, thickness=2)

        # Print the length of longest line found
        length = (((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** 0.5
        print(f'The length of the longest {output_folder_name} line is ({length})')

    # Write longest line image to the output folder
    cv2.imwrite(f'output\\{output_folder_name}\\06_longest_line_image.png', longest_line_image)

    # Remove all found lines from the image
    no_lines_image = thresholded_image - all_lines_image

    # Write the thresholded image without the found lines to the output folder
    cv2.imwrite(f'output\\{output_folder_name}\\07_no_lines_image.png', no_lines_image)

    # Apply erosion and then dilation on the image to eliminate small random noise pixels
    kernel = np.ones((3, 3), 'uint8')
    eroded_image = cv2.erode(no_lines_image, kernel)
    dilated_image = cv2.dilate(eroded_image, kernel)

    # Write the eroded image and the dilated image to the output folder
    cv2.imwrite(f'output\\{output_folder_name}\\08_eroded_image.png', eroded_image)
    cv2.imwrite(f'output\\{output_folder_name}\\09_dilated_image.png', dilated_image)

    # Find the circles in the image
    circles = cv2.HoughCircles(dilated_image, method=cv2.HOUGH_GRADIENT, dp=1, minDist=100, param2=22)
    circles = np.uint16(np.around(circles))

    # Draw all found circles
    all_circles = np.zeros(image_dimensions, 'uint8')
    for circle in circles:
        for x, y, radius in circle:
            cv2.circle(all_circles, center=(x, y), radius=radius, color=255, thickness=1)

    # Write all the found circles image to the output folder
    cv2.imwrite(f'output\\{output_folder_name}\\10_all_circles.png', all_circles)

    # Find the largest circle
    largest_circle = get_largest_circle(circles)

    # Fill and draw the largest circle on the final image
    for x, y, radius in largest_circle:
        largest_circle_image = np.zeros(image_dimensions, 'uint8')
        cv2.circle(largest_circle_image, (x, y), radius, 255, thickness=1)
        filled_circle_image = fill_hole(largest_circle_image, known_pixel=(y, x))
        filled_circle_image = np.repeat(filled_circle_image, 3)
        filled_circle_image = np.reshape(filled_circle_image, image.shape)
        colored_filled_circle_image = np.where(filled_circle_image == (255, 255, 255),
                                               np.full(image.shape, largest_circle_color), filled_circle_image)
        final_image = np.where(filled_circle_image == (255, 255, 255), colored_filled_circle_image, final_image)

        # Print the radius and center of the largest circle found:
        print(f'The center of the largest {output_folder_name} circle is {(x, y)} and its radius is ({radius})')

        # Write the largest found circle image to the output folder
        cv2.imwrite(f'output\\{output_folder_name}\\11_largest_circle_image.png', largest_circle_image)

    print()
    return


# Read the image
image = cv2.imread("input\\test.jpg")
height, width, _ = image.shape
image_dimensions = (height, width)
final_image = np.full(image.shape, (100, 100, 100))

# Remove random white spots in the image
white_suppressed_image = replace_color_with(image, center=[255, 255, 255], radius=250, color=[0, 0, 0])

# Write image after white spots removal to the output folder
cv2.imwrite('output\\01_white_suppressed_image.png', white_suppressed_image)

# Blur the image to decrease noise variance
median_blurred_image = cv2.medianBlur(white_suppressed_image, ksize=7)

# Write median-blurred image to the output folder
cv2.imwrite('output\\02_median_blurred_image.png', median_blurred_image)

# Split the image into the three main RGB colors
blue, green, red = cv2.split(median_blurred_image)

# Find the desired objects (Largest circle, and longest line)
# Note that the colors are represented as BGR instead of RGB as the default color space for cv2
find_objects(red, largest_circle_color=(0, 0, 255), longest_line_color=(0, 0, 0), output_folder_name='red')
find_objects(green, largest_circle_color=(0, 255, 0), longest_line_color=(255, 0, 255), output_folder_name='green')
find_objects(blue, largest_circle_color=(255, 0, 0), longest_line_color=(0, 255, 255), output_folder_name='blue')

# Write the image to the output folder
cv2.imwrite('output\\final_image.png', final_image)
