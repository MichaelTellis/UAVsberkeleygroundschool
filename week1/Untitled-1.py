import cv2
import numpy as np
import matplotlib.pyplot as plt

filepath = "/Users/michaeltellis/BerkeleyClasses/UAVSBerkeley/UAVsberkeleygroundschool/test.png"
img = cv2.imread(filepath)
half = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
bigger = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
stretch_near = cv2.resize(img, (780, 540),
            interpolation = cv2.INTER_NEAREST)

rows, cols = img.shape[:2]
translation_matrix = np.float32([[1, 0, 100], [0, 1, 50]])
rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
print(rows, cols)
rotated = cv2.warpAffine(img, translation_matrix, (cols, rows))
edges = cv2.Canny(img, 100, 200)
cv2.imshow("edges", edges)
cv2.imshow('Rotate Image', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
cv2.imshow('Original Image', half)
cv2.imshow('Bigger Image', bigger)
cv2.imshow('Stretch Nearest', stretch_near)
cv2.imshow('Rotate Image', rotate)



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cv2.imshow('Gray Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()



if img is None:
    print("Error: Could not load the image.")
else:
    filename = "savedImage.jpg"
    cv2.imwrite(filename, gray)
    print(f"Image saved as {filename}")
'''