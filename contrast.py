import cv2
import numpy as np

params = cv2.SimpleBlobDetector_Params()

kernel = np.ones((2,2), np.uint8)

img = cv2.imread('corneal1.png',0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

dst = cv2.fastNlMeansDenoising(cl1)

#dil = cv2.dilate(dst,kernel, iterations=2)

blur = cv2.bilateralFilter(dst, 5, 75, 75)

thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)

erosion = cv2.erode(thresh, kernel, iterations=0)

#lines = cv2.HoughLines(dil, 2, 45, 10)

#median = cv2.medianBlur(dst,9)

params.minThreshold = 0.5
params.maxThreshold = 100

# Filter by Area.
params.filterByArea = True
params.minArea = 0.5

# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.4

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(erosion)

imk = cv2.drawKeypoints(erosion, keypoints, np.array([]), (0,255,0))

cv2.imwrite('res.jpg',imk)


