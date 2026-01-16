from config import DATA_DIR, OUTPUT_DIR
import os
import numpy as np
import cv2

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

def rotation_matrix_to_euler(R):
    """
    Convert 3x3 rotation matrix to Euler angles (yaw, pitch, roll).
    Uses ZYX convention: yaw (Z), pitch (Y), roll (X).
    
    Returns:
        tuple: (yaw, pitch, roll) in degrees
    """

    # Handle gimbal lock
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    # Not at gimbal lock
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])

    # Gimbal lock
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    
    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

def produce_motions(path1="sample1.jpg", path2="sample2.jpg"):
    # Configure image paths
    img1_path= os.path.join(DATA_DIR, path1)
    img2_path= os.path.join(DATA_DIR, path2)

    # Read images as grayscale
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess
    processed_img1 = preprocess_image(img1)
    processed_img2 = preprocess_image(img2)

    sift = cv2.SIFT_create()
    kp1, kp2, good_matches = find_matches(sift, processed_img1, processed_img2)

    print(f"Image 1: {len(kp1)} keypoints")
    print(f"Image 2: {len(kp2)} keypoints")
    print(f"Good matches: {len(good_matches)}")

    # Draw and save matches
    img_matches = cv2.drawMatches(img1, kp1,img2, kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    output_path = os.path.join(OUTPUT_DIR, "sift_matches.jpg")
    cv2.imwrite(output_path, img_matches)
    print(f"Saved to {output_path}")

    # Produce transformation
    H = get_transformation(kp1, kp2, good_matches)
    if H is None:
        return None
    
    # Decompose homography to get rotation matrix
    # Camera intrinsic matrix (identity if unknown)
    K = np.eye(3)
    _, Rs, Ts, normals = cv2.decomposeHomographyMat(H, K)

    # Produce rotation matrix and invert for camera motion
    R = Rs[0]
    R_camera = R.T

    # Convert to Eulerian angles
    return rotation_matrix_to_euler(R_camera)