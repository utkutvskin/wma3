import cv2
import numpy as np


template_img = cv2.imread('photo_3_train.jpg', cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture('video_3_query.mp4')

sift = cv2.SIFT_create()

# Compute keypoints and descriptors for the template image once
kp_template, des_template = sift.detectAndCompute(template_img, None)

# Setup FLANN matcher
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute keypoints and descriptors for the current video frame
    kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)

    # Proceed only if descriptors are found in the frame
    if des_frame is not None and len(des_frame) > 2:
        matches = flann.knnMatch(des_template, des_frame, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        # Proceed with homography if enough robust matches are found
        if len(good_matches) > 8:
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Calculate Homography matrix using RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = template_img.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                # Transform the bounding box coordinates
                dst = cv2.perspectiveTransform(pts, M)

                # Draw a red bounding polygon around the tracked object in the frame
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('Program 3 - Dino Video Tracking', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()