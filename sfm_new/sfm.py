#!/usr/bin/env python3
"""
Structure from Motion 3D Reconstruction

This script reconstructs a sparse 3D point cloud from a sequence of images using the following pipeline:

1. Preprocessing and Initialization:
    - Loads all images (supports JPG and PNG) from a user-defined folder.
    - Optionally downsamples the images to reduce computation cost.
    - Uses the first two images for initialization.

2. Feature Extraction & Matching:
    - Uses SIFT to extract keypoints and descriptors.
    - Matches features using a FLANN-based KNN matcher and applies Lowe’s ratio test.

3. Relative Pose Estimation:
    - Computes the Essential Matrix from the matched keypoints.
    - Decomposes the Essential Matrix to recover the relative rotation and translation.

4. Triangulation & Initial 3D Point Cloud Construction:
    - Triangulates matched keypoints to build an initial 3D point cloud.
    - Computes the reprojection error to assess accuracy.

5. Iterative PnP Registration and Point Cloud Update:
    - For each subsequent image, finds common keypoints, estimates the new camera pose via PnP,
      triangulates both common and new points, and accumulates them into the point cloud.

6. Final Output:
    - Saves the reconstructed 3D point cloud (with pixel colors) to a PLY file.

Usage:
    python sfm.py --input_folder /path/to/images --downsample 2.0 --output sparse.ply [--K /path/to/calibration.txt]
"""

import os
import cv2
import glob
import numpy as np
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Structure from Motion 3D Reconstruction')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to folder containing input images (jpg, png)')
    parser.add_argument('--downsample', type=float, default=2.0,
                        help='Downsampling scale factor (default: 2.0, e.g., 2.0 means one pyrDown step)')
    parser.add_argument('--output', type=str, default='sparse.ply',
                        help='Output PLY file name (default: sparse.ply)')
    parser.add_argument('--K', type=str, default=None,
                        help='Path to a txt file containing a 3x3 camera calibration matrix')
    return parser.parse_args()

def subsample(image, down_sample_factor):
    """
    Downsamples the image by the given factor (assumes factor is a power of 2).
    """
    try:
        # Compute number of pyrDown steps (log2(down_sample_factor))
        steps = int(round(np.log2(down_sample_factor)))
        for _ in range(steps):
            image = cv2.pyrDown(image)
        return image
    except Exception as e:
        raise Exception("Error in subsampling image: " + str(e))

def triangulation(proj_ref, proj_next, kp1, kp2):
    """
    Triangulates matched keypoints given two projection matrices.
    Internally, keypoints are transposed for OpenCV's triangulation function,
    but the original (N,2) arrays are returned.
    """
    try:
        kp1_t = kp1.T  # shape: (2, N)
        kp2_t = kp2.T  # shape: (2, N)
        cloud = cv2.triangulatePoints(proj_ref, proj_next, kp1_t, kp2_t)
        cloud /= cloud[3]
        return kp1, kp2, cloud
    except Exception as e:
        raise Exception("Error in triangulation: " + str(e))

def perspective_n_point(cloud, kp1, kp2, K):
    """
    Estimates the camera pose using the Perspective-n-Point (PnP) algorithm.
    Input keypoints (kp1 and kp2) are expected to be of shape (N, 2).
    Returns rotation, translation, and the inlier keypoints and corresponding 3D points.
    """
    try:
        dist_coeff = np.zeros((5, 1))
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(cloud, kp1, K, dist_coeff, flags=cv2.SOLVEPNP_ITERATIVE)
        inliers = inliers.flatten()
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec, kp2[inliers], cloud[inliers], kp1[inliers]
    except Exception as e:
        raise Exception("Error in perspective_n_point: " + str(e))

def reprojection_error(cloud, kp, trans_mat, K):
    """
    Projects 3D points back onto the image plane and computes the reprojection error.
    The input 'kp' is expected to be of shape (N, 2) and the projected points will be (N, 2).
    """
    try:
        R, _ = cv2.Rodrigues(trans_mat[:3, :3])
        kp_reprojected, _ = cv2.projectPoints(cloud, R, trans_mat[:3, 3], K, distCoeffs=None)
        kp_reprojected = np.float32(kp_reprojected[:, 0, :])  # shape: (N, 2)
        # Optional check: ensure shapes match
        if kp_reprojected.shape != kp.shape:
            raise Exception(f"Shape mismatch: reprojected {kp_reprojected.shape} vs original {kp.shape}")
        error = cv2.norm(kp_reprojected, kp, cv2.NORM_L2)
        return error / len(kp_reprojected), cloud, kp_reprojected
    except Exception as e:
        raise Exception("Error in reprojection_error: " + str(e))

def three_view_points(kp2, kp2_dash, kp3):
    """
    Identifies keypoints common to three consecutive views and those unique to the new view.
    
    Parameters:
      kp2: keypoints from image n-1 to n.
      kp2_dash: keypoints from image n to n+1 (first set).
      kp3: keypoints from image n to n+1 (second set).
    
    Returns:
      index1: indices of keypoints common in the first two sets.
      kp2_dash_common: common keypoints from kp2_dash.
      kp3_common: common keypoints from kp3.
      kp2_dash_uncommon, kp3_uncommon: keypoints unique to the new view.
    """
    try:
        index1 = []
        index2 = []
        for i in range(kp2.shape[0]):
            # Check if the keypoint in kp2 appears in kp2_dash (element-wise comparison)
            if (kp2[i, :] == kp2_dash).any():
                index1.append(i)
            x = np.where(kp2_dash == kp2[i, :])
            if x[0].size != 0:
                index2.append(x[0][0])
        # Identify keypoints not common in the new view
        kp3_uncommon = []
        kp2_dash_uncommon = []
        for k in range(kp3.shape[0]):
            if k not in index2:
                kp3_uncommon.append(list(kp3[k, :]))
                kp2_dash_uncommon.append(list(kp2_dash[k, :]))
        index1 = np.array(index1)
        index2 = np.array(index2)
        kp2_dash_common = kp2_dash[index2]
        kp3_common = kp3[index2]
        return index1, kp2_dash_common, kp3_common, np.array(kp2_dash_uncommon), np.array(kp3_uncommon)
    except Exception as e:
        raise Exception("Error in three_view_points: " + str(e))

def get_features(image1, image2):
    """
    Extracts SIFT features from two images and returns matched keypoints.
    The returned keypoints are in shape (N, 2).
    """
    try:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # Use SIFT (compatible with OpenCV >=4.4; fallback to xfeatures2d if needed)
        try:
            sift = cv2.SIFT_create()
        except Exception:
            sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        des1 = np.float32(des1)
        des2 = np.float32(des2)
        # FLANN-based matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10) # change this to get better results (default trees=5)
        search_params = dict(checks=200) # change this to get better results (default checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        filtered_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                filtered_matches.append(m)
        kp1 = np.float32([kp1[m.queryIdx].pt for m in filtered_matches])
        kp2 = np.float32([kp2[m.trainIdx].pt for m in filtered_matches])
        return kp1, kp2
    except Exception as e:
        raise Exception("Error in get_features: " + str(e))

def output(final_cloud, pixel_colour, output_filename):
    """
    Saves the reconstructed 3D point cloud (with colors) to a PLY file.
    """
    try:
        # Scale the 3D points for visualization (adjust the factor as needed)
        output_points = final_cloud.reshape(-1, 3) * 200 # default 200  
        output_colors = pixel_colour.reshape(-1, 3)
        mesh = np.hstack([output_points, output_colors])
        # Optionally, clean the point cloud by removing outliers
        mesh_mean = np.mean(mesh[:, :3], axis=0)
        diff = mesh[:, :3] - mesh_mean
        distance = np.sqrt(np.sum(diff**2, axis=1))
        index = np.where(distance < np.mean(distance) + 300)
        mesh = mesh[index]
        ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar blue
property uchar green
property uchar red
end_header
'''
        with open(output_filename, 'w') as f:
            f.write(ply_header % dict(vert_num=len(mesh)))
            np.savetxt(f, mesh, fmt='%f %f %f %d %d %d')
        print("Point cloud processed, cleaned and saved successfully to", output_filename)
    except Exception as e:
        raise Exception("Error in output function: " + str(e))

def main():
    args = parse_args()
    
    # --------------------------------------------------
    # 1. Dataset Preparation: Load images from folder
    # --------------------------------------------------
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_folder, ext)))
    if len(image_paths) < 2:
        print("Error: Need at least 2 images for initialization.")
        sys.exit(1)
    image_paths = sorted(image_paths)
    
    dataset = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Unable to read image {path}")
            sys.exit(1)
        dataset.append(img)
    
    # --------------------------------------------------
    # 2. Camera Intrinsics Setup
    # --------------------------------------------------
    down_sample = args.downsample
    # If a calibration file is provided, load K from the file
    if args.K is not None:
        try:
            K = np.loadtxt(args.K)
            if K.shape != (3, 3):
                raise ValueError("Calibration matrix must be 3x3.")
            # Adjust K for the downsampling factor
            K[0,0] /= down_sample
            K[1,1] /= down_sample
            K[0,2] /= down_sample
            K[1,2] /= down_sample
            print("Loaded calibration matrix K from file:")
        except Exception as e:
            print(f"Error loading calibration matrix from {args.K}: {e}")
            sys.exit(1)
    else:
        # Use default intrinsics computed from the first image dimensions.
        h, w, _ = dataset[0].shape
        fx = w
        fy = w  # assuming square pixels
        cx = w / 2
        cy = h / 2
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        # Adjust K for the downsampling factor
        K[0,0] /= down_sample
        K[1,1] /= down_sample
        K[0,2] /= down_sample
        K[1,2] /= down_sample
        print("Using default computed calibration matrix K:")
    
    print(K)
    
    # --------------------------------------------------
    # 3. Initialization
    # --------------------------------------------------
    initial = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]])
    proj_1 = K @ initial
    trans_12 = np.empty((3, 4))
    proj_ref = proj_1
    final_cloud = np.zeros((1, 3))
    pixel_colour = np.zeros((1, 3))
    
    try:
        # Downsample the first two images to reduce computation
        image1 = subsample(dataset[0], down_sample)
        image2 = subsample(dataset[1], down_sample)
    except Exception as e:
        print(e)
        sys.exit(1)
    
    # --------------------------------------------------
    # 4. Feature Extraction & Matching (Initialization)
    # --------------------------------------------------
    try:
        kp1, kp2 = get_features(image1, image2)
    except Exception as e:
        print(e)
        sys.exit(1)
    
    # --------------------------------------------------
    # 5. Relative Pose Estimation Between the First Two Images
    # --------------------------------------------------
    try:
        E, mask = cv2.findEssentialMat(kp1, kp2, K, method=cv2.RANSAC, prob=0.999, threshold=0.4)
        kp1 = kp1[mask.ravel() == 1]
        kp2 = kp2[mask.ravel() == 1]
    except Exception as e:
        print("Error in Essential Matrix computation:", e)
        sys.exit(1)
    
    try:
        _, rot, trans, mask_pose = cv2.recoverPose(E, kp1, kp2, K)
        trans = trans.ravel()
        trans_12[:3, :3] = rot @ initial[:3, :3]
        trans_12[:3, 3] = initial[:3, 3] + (initial[:3, :3] @ trans)
        proj_2 = K @ trans_12
        # Optionally filter keypoints using mask_pose
        kp1 = kp1[mask_pose.ravel() > 0]
        kp2 = kp2[mask_pose.ravel() > 0]
    except Exception as e:
        print("Error in recovering pose:", e)
        sys.exit(1)
    
    # --------------------------------------------------
    # 6. Triangulation & Initial 3D Point Cloud Construction
    # --------------------------------------------------
    try:
        kp1, kp2, cloud = triangulation(proj_1, proj_2, kp1, kp2)
        # Convert from homogeneous coordinates
        cloud = cv2.convertPointsFromHomogeneous(cloud.T)
        # Call reprojection_error with keypoints in shape (N,2) (no transpose)
        error, cloud, repro_pts = reprojection_error(cloud, kp2, trans_12, K)
        print("Reprojection Error after 2 images:", np.round(error, 4))
        # Initiate the PnP pipeline (for debugging; result not used further here)
        R_pnp, t_pnp, kp2_valid, cloud, kp1_valid = perspective_n_point(cloud[:, 0, :], kp1, kp2, K)
    except Exception as e:
        print("Error during initial triangulation and PnP:", e)
        sys.exit(1)
    
    # --------------------------------------------------
    # 7. Iterative PnP Registration and Point Cloud Update
    # --------------------------------------------------
    for i in range(len(dataset) - 2):
        try:
            if i > 0:
                # Update reference by re-triangulating the previous matches
                kp1, kp2, cloud = triangulation(proj_1, proj_2, kp1, kp2)
                # Convert from homogeneous coordinates and extract 3D points
                cloud = cv2.convertPointsFromHomogeneous(cloud.T)
                cloud = cloud[:, 0, :]
            new_image = dataset[i + 2]
            new_image = subsample(new_image, down_sample)
            kp2_dash, kp3 = get_features(image2, new_image)
            
            # Identify keypoints common to the three views
            index, kp2_dash_common, kp3_common, kp2_dash_uncommon, kp3_uncommon = three_view_points(kp2, kp2_dash, kp3)
            
            # Estimate camera pose using the common keypoints via PnP
            R, t, kp3_common, cloud_common, kp2_dash_common = perspective_n_point(cloud[index], kp3_common, kp2_dash_common, K)
            trans_mat_new = np.hstack((R, t))
            proj_new = K @ trans_mat_new
            error1, cloud_common, kp_projected = reprojection_error(cloud_common, kp3_common, trans_mat_new, K)
            
            # Triangulate the new (non-common) points
            kp2_dash_uncommon, kp3_uncommon, cloud_new = triangulation(proj_2, proj_new, kp2_dash_uncommon, kp3_uncommon)
            cloud_new = cv2.convertPointsFromHomogeneous(cloud_new.T)
            error2, cloud_new, kp_reprojected = reprojection_error(cloud_new, kp3_uncommon, trans_mat_new, K)
            
            print("Reprojection Error after " + str(i + 3) + " images:", np.round(error1 + error2, 4))
            
            # Stack the new 3D points into the final point cloud
            final_cloud = np.vstack((final_cloud, cloud_new[:, 0, :]))
            
            # Retrieve pixel colors from the new image at the keypoint locations
            kp_for_intensity = np.array(kp3_uncommon, dtype=np.int32)
            colors = []
            for intensity in kp_for_intensity:
                x, y = intensity[0], intensity[1]
                if y < new_image.shape[0] and x < new_image.shape[1]:
                    colors.append(new_image[y, x])
                else:
                    colors.append([0, 0, 0])
            colors = np.array(colors)
            pixel_colour = np.vstack((pixel_colour, colors))
            
            # Update variables for the next iteration
            proj_1 = proj_2
            proj_2 = proj_new
            image2 = new_image
            kp1 = kp2_dash
            kp2 = kp3
        except Exception as e:
            print(f"Error during processing image index {i + 2}: {e}")
    
    # --------------------------------------------------
    # 8. Final Output: Save the point cloud to a PLY file
    # --------------------------------------------------
    try:
        output(final_cloud, pixel_colour, args.output)
    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main()
