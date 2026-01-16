from config import DATA_DIR
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

def preprocess_image(img):
    """preprocessing to improve feature matching."""

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    
    # Normalize and apply gaussian blur
    img_norm = cv2.normalize(img_clahe, None, 0, 255, cv2.NORM_MINMAX)
    img_blur = cv2.GaussianBlur(img_norm, (3, 3), 0)
    
    # Enhance edges
    img_edges = cv2.Laplacian(img_blur, cv2.CV_8U, ksize=3)
    img_enhanced = cv2.addWeighted(img_blur, 0.7, img_edges, 0.3, 0)
    
    return img_enhanced

def find_matches(detector, img1, img2):
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        print("Warning: No features detected in one or both images")
        return None, None, []

    # Match descriptors using BFMatch 
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good_matches.append(m)

    return kp1, kp2, good_matches

def valid_homography(H):
    """Check if homography matrix is valid."""
    if H is None:
        return False
        
    # Check if matrix is finite
    if not np.all(np.isfinite(H)):
        return False
        
    # Check determinant is positive and not too close to zero
    det = np.linalg.det(H)
    if det < 1e-6:
        return False
    
    return True

def validate_transformation(src_pts, dst_pts, H):
    """Validate transformation by checking reprojection error."""
    if len(src_pts) < 4:
        return False, float('inf')
        
    # Project points using homography
    transformed_pts = cv2.perspectiveTransform(src_pts, H)
    
    # Calculate reprojection error
    errors = np.sqrt(np.sum((dst_pts - transformed_pts) ** 2, axis=2))
    
    # Remove outliers before computing mean
    sorted_errors = np.sort(errors.flatten())
    inlier_count = int(len(sorted_errors) * 0.8)
    if inlier_count > 0:
        inlier_errors = sorted_errors[:inlier_count]
        mean_error = np.mean(inlier_errors)
    else:
        mean_error = np.mean(errors)
    
    return float(mean_error)

def get_transformation(kp1, kp2, matches):
    """Get transformation from keypoints and matches."""
    if len(matches) < 4:
        print("Warning: Not enough matches to compute transformation")
        return None
        
    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    methods = [
        (cv2.RANSAC, 3.0),
        (cv2.RANSAC, 5.0),
        (cv2.LMEDS, 0),
        (cv2.RHO, 5.0)
    ]
    best_H = None
    best_error = float('inf')

    # Try multiple methods with varying params 
    for method, param in methods:
        H, mask = cv2.findHomography(src_pts, dst_pts, method, param)
        
        if H is not None and valid_homography(H):
            # Check reprojection error
            error = validate_transformation(src_pts, dst_pts, H)
            
            if error < best_error:
                best_H = H
                best_error = error
    
    return best_H

def process_frame(img1, img2):
    # Preprocess
    processed_img1 = preprocess_image(img1)
    processed_img2 = preprocess_image(img2)

    sift = cv2.SIFT_create()
    kp1, kp2, good_matches = find_matches(sift, processed_img1, processed_img2)

    if len(good_matches) < 10:
        return (0.0, 0.0, 0.0)

    # Get matched point pairs
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Image center
    img_width = img1.shape[1]
    img_height = img1.shape[0]
    cx, cy = img_width / 2, img_height / 2
    
    # Compute displacement
    dx = pts2[:, 0] - pts1[:, 0]
    dy = pts2[:, 1] - pts1[:, 1]
    
    # Use median for translation (yaw / pitch)
    median_dx = np.median(dx)
    median_dy = np.median(dy)
    
    # Remove translation to isolate rotation
    pts2_corrected = pts2.copy()
    pts2_corrected[:, 0] -= median_dx
    pts2_corrected[:, 1] -= median_dy
    
    # Compute roll from the residual rotational motion around image center
    angles1 = np.arctan2(pts1[:, 1] - cy, pts1[:, 0] - cx)
    angles2 = np.arctan2(pts2_corrected[:, 1] - cy, pts2_corrected[:, 0] - cx)
    
    # Angular difference
    angle_diff = angles2 - angles1
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
    
    # Median angular change is the roll
    median_roll = np.degrees(np.median(angle_diff))
    
    # Convert pixel shift to approximate rotation angle with ~60 degree FOV
    h_fov = 60.0
    v_fov = h_fov * img_height / img_width
    
    pixels_per_degree_h = img_width / h_fov
    pixels_per_degree_v = img_height / v_fov

    # Compute final angles 
    yaw = median_dx / pixels_per_degree_h
    pitch = -median_dy / pixels_per_degree_v
    roll = -median_roll
    
    return (yaw, roll, pitch)

def video_to_motions(path='sample.mov'):
    video_path = os.path.join(DATA_DIR, path)
    vidcap = cv2.VideoCapture(video_path)

    prev_frame = None
    motions = []
    while True:
        success, frame = vidcap.read()

        # End of video
        if not success:
            break
            
        # Process motion
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            motions.append(process_frame(prev_frame, gray_frame))
        prev_frame = gray_frame

    vidcap.release()
    cv2.destroyAllWindows()

    return motions