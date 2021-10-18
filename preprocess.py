import cv2 as cv
import numpy as np
from skimage import img_as_ubyte

DESIRED_SIZE = (512, 512)


def show(image):
    cv.imshow('w', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def extract_resized_image(image):
    # Extract green channel
    # Apply log threshold with constant multiplier of 5
    # Convert back to CV_8U
    # Extract the threshold given by OTSU algorithm
    green = image[:, :, 1]
    log_transform = np.uint8(5 * np.log(cv.add(1, green)))
    ret, thresh = cv.threshold(log_transform, 0, 255, cv.THRESH_OTSU)

    # Find largest contour in the thresholded image. Find the bounding rectangle for it
    _, contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(c)
    desired_size = max(w, h)

    # Draw the largest contour, filled, on a black mask and use that to mask the image
    mask = np.zeros_like(green)
    cv.drawContours(mask, [c], 0, 255, -1)
    masked_image = cv.bitwise_and(image, cv.cvtColor(mask, cv.COLOR_GRAY2BGR))

    # Crop the masked image, draw borders to make it square
    cropped_image = masked_image[y:y + h, x:x + w]

    new_size = cropped_image.shape[:2]
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2

    final_image = cv.copyMakeBorder(cropped_image, top, bottom, left, right,
                                    cv.BORDER_CONSTANT, value=(0, 0, 0))
    final_image = cv.resize(final_image, DESIRED_SIZE, interpolation=cv.INTER_CUBIC)
    return final_image


def preprocess(image, resize=True):
    if resize:
        image = extract_resized_image(image)
    b = np.zeros(image.shape[:2], np.uint8)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    b[gray != 0] = 255
    b = cv.cvtColor(b, cv.COLOR_GRAY2BGR)
    b = np.float64(b) / 255
    aa = cv.addWeighted(image, 4, cv.GaussianBlur(image, (0, 0), 256 / 30.0), -4, 128) * b + 128 * (1 - b)
    b = cv.erode(b, np.ones((15, 15), np.uint8))
    c = np.zeros(image.shape[:2], np.uint8)
    cv.circle(c, (256, 256), 245, (1, 1, 1), -1, 8, 0)
    c[b[:, :, 2] == 0] = 0
    aa[c == 0] = 0
    return np.uint8(aa)


if __name__ == '__main__':

    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir')
    parser.add_argument('--result_dir')
    args = parser.parse_args()

    imgs = os.listdir(args.image_dir)
    for i, f in enumerate(imgs):
        img = cv.imread(os.path.join(args.image_dir, f))
        prep = preprocess(img)
        cv.imwrite(os.path.join(args.result_dir, f.split('.')[0] + '.png'), prep)
        # cv.imshow('w', prep), cv.waitKey(0), cv.destroyAllWindows()
