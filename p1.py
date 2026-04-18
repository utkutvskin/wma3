import cv2
import numpy as np

img = cv2.imread('photo_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Part 1
gray_float = np.float32(gray)

# Compute Harris response matrix
harris_response = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)

# Dilate to make local maxima more distinct
harris_response = cv2.dilate(harris_response, None)

# Find top 4 corners with non-maximum suppression
min_distance = 150
top_corners = []
response_copy = harris_response.copy()

for _ in range(4):
    # Find the location of the maximum response
    _, max_val, _, max_loc = cv2.minMaxLoc(response_copy)
    top_corners.append(max_loc)

    # Suppress a region around this corner so next iteration picks a different area
    x, y = max_loc
    cv2.circle(response_copy, (x, y), min_distance, 0, -1)

img_harris = img.copy()
for x, y in top_corners:
    cv2.circle(img_harris, (x, y), 8, (0, 0, 255), -1)

cv2.imshow('Program 1 - Top 4 Harris Corners', img_harris)

# --- Part 2: SIFT Keypoints ---
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)
img_sift = cv2.drawKeypoints(gray, kp, img.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Program 1 - SIFT Keypoints', img_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()