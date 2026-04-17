import cv2
import numpy as np

# Load query (template) and train (scene) images in grayscale
img_query = cv2.imread('photo_2_query.jpg', cv2.IMREAD_GRAYSCALE)
img_train = cv2.imread('photo_2_train.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT and extract keypoints and descriptors
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_query, None)
kp2, des2 = sift.detectAndCompute(img_train, None)

# Setup FLANN matcher with KD-Tree algorithm
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Perform KNN matching (k=2)
matches = flann.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test to filter out false matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

MIN_MATCH_COUNT = 10
img_train_color = cv2.imread('photo_2_train.jpg')

if len(good_matches) > MIN_MATCH_COUNT:
    # Extract coordinates of the good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculate Homography matrix using RANSAC to exclude outliers
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Define corners of the query image
    h, w = img_query.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    # Project the corners onto the train image to find the object boundary
    if M is not None:
        dst = cv2.perspectiveTransform(pts, M)
        # Draw a green bounding polygon around the matched object
        img_train_color = cv2.polylines(img_train_color, [np.int32(dst)], True, (0, 255, 0), 5, cv2.LINE_AA)
else:
    print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
    matchesMask = None

# Draw only the inlier matches
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
img_result = cv2.drawMatches(img_query, kp1, img_train_color, kp2, good_matches, None, **draw_params)

# Resize the output image to fit the screen
scale_percent = 60
width = int(img_result.shape[1] * scale_percent / 100)
height = int(img_result.shape[0] * scale_percent / 100)
img_result_resized = cv2.resize(img_result, (width, height))

cv2.imshow('Program 2 - Homography Feature Matching', img_result_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()