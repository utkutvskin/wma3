import cv2
import numpy as np

# Load the image and convert it to grayscale
img = cv2.imread('photo_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Part 1: Harris Corner Detection (Top 4 Corners) ---
# Detect the 4 strongest corners using the Harris detector backend
corners = cv2.goodFeaturesToTrack(gray, maxCorners=4, qualityLevel=0.01, minDistance=10, useHarrisDetector=True, k=0.04)
corners = np.int32(corners)

img_harris = img.copy()
# Draw a red circle around each detected corner
for i in corners:
    x, y = i.ravel()
    cv2.circle(img_harris, (x, y), 8, (0, 0, 255), -1)

cv2.imshow('Program 1 - Top 4 Harris Corners', img_harris)

# --- Part 2: SIFT Keypoints ---
# Initialize SIFT detector and find keypoints
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)

# Draw keypoints with their size and orientation features
img_sift = cv2.drawKeypoints(gray, kp, img.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Program 1 - SIFT Keypoints', img_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()