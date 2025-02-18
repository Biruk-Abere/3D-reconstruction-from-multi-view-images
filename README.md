## 1. Dataset Preparation

**Objective:**  
Load a sequence of images from a user-specified folder, sort them, and optionally downsample to reduce computational load.

**Details:**

- **Image Loading:**  
  The code uses the `glob` module to gather all images with extensions like `.jpg`, `.jpeg`, and `.png` from the given folder.  
  ```python
  image_extensions = ['*.jpg', '*.jpeg', '*.png']
  image_paths = []
  for ext in image_extensions:
      image_paths.extend(glob.glob(os.path.join(args.input_folder, ext)))
  image_paths = sorted(image_paths)
  ```
- **Downsampling:**  
  Downsampling is performed using OpenCV's `pyrDown` function to reduce the image resolution. This helps lower the computational cost during feature detection and matching.
  ```python
  def subsample(image, down_sample_factor):
      steps = int(round(np.log2(down_sample_factor)))
      for _ in range(steps):
          image = cv2.pyrDown(image)
      return image
  ```
- **Sorting:**  
  Sorting the image paths ensures a consistent sequential order which is crucial for the reconstruction process.

---

## 2. Camera Intrinsics Setup

**Objective:**  
Define the camera intrinsic matrix, which is necessary for projecting 3D points into 2D image coordinates.

**Details:**

- **Default Calibration:**  
  When calibration data is not available, a heuristic is used:
  - **Focal Lengths (`fx` and `fy`):** Set equal to the image width.
  - **Principal Point (`cx`, `cy`):** Assumed to be at the center of the image.
  ```python
  h, w, _ = dataset[0].shape
  fx = w
  fy = w  # Assuming square pixels
  cx = w / 2
  cy = h / 2
  K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]], dtype=np.float64)
  ```
- **Adjustment for Downsampling:**  
  The intrinsic matrix is scaled according to the downsampling factor.
  ```python
  K[0,0] /= down_sample
  K[1,1] /= down_sample
  K[0,2] /= down_sample
  K[1,2] /= down_sample
  ```
- **Why This Matters:**  
  A correct intrinsic matrix is vital because all subsequent computations (e.g., essential matrix estimation, triangulation, reprojection) depend on accurately mapping 3D points to the image plane.

---

## 3. Initialization

**Objective:**  
Set up the initial projection matrix and baseline reconstruction using the first two images.

**Details:**

- **Initial Projection Matrix:**  
  The projection matrix for the first image is constructed by multiplying the intrinsic matrix `K` with an initial pose (identity rotation and zero translation).
  ```python
  initial = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]])
  proj_1 = K @ initial
  ```
- **Reference Variables:**  
  Variables such as `final_cloud` (to store 3D points) and `pixel_colour` (to store corresponding colors) are initialized.
- **Downsampling First Two Images:**  
  The first two images are downsampled before feature extraction:
  ```python
  image1 = subsample(dataset[0], down_sample)
  image2 = subsample(dataset[1], down_sample)
  ```

---

## 4. Feature Extraction & Matching

**Objective:**  
Detect features in the images and establish correspondences between the first two images.

**Details:**

- **SIFT Feature Extraction:**  
  SIFT is used to extract keypoints and descriptors. This provides scale and rotation invariant features.
  ```python
  def get_features(image1, image2):
      gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
      gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
      try:
          sift = cv2.SIFT_create()
      except Exception:
          sift = cv2.xfeatures2d.SIFT_create()
      kp1, des1 = sift.detectAndCompute(gray1, None)
      kp2, des2 = sift.detectAndCompute(gray2, None)
      ...
  ```
- **FLANN-based Matching and Lowe’s Ratio Test:**  
  FLANN is used for fast approximate matching of feature descriptors. Lowe’s ratio test filters out ambiguous matches.
  ```python
  matches = flann.knnMatch(des1, des2, k=2)
  for m, n in matches:
      if m.distance < 0.7 * n.distance:
          filtered_matches.append(m)
  ```
- **Output:**  
  Returns keypoints in a consistent `(N, 2)` format for further processing.

---

## 5. Relative Pose Estimation

**Objective:**  
Estimate the relative rotation and translation between the first two views.

**Details:**

- **Essential Matrix Computation:**  
  The Essential Matrix, encoding the epipolar geometry, is computed using RANSAC.
  ```python
  E, mask = cv2.findEssentialMat(kp1, kp2, K, method=cv2.RANSAC, prob=0.999, threshold=0.4)
  ```
- **Pose Recovery:**  
  The Essential Matrix is decomposed to obtain the relative camera rotation and translation.
  ```python
  _, rot, trans, mask_pose = cv2.recoverPose(E, kp1, kp2, K)
  ```
- **Projection Matrix Update:**  
  A new projection matrix for the second image is computed:
  ```python
  trans_12[:3, :3] = rot @ initial[:3, :3]
  trans_12[:3, 3] = initial[:3, 3] + (initial[:3, :3] @ trans)
  proj_2 = K @ trans_12
  ```

---

## 6. Triangulation & Initial 3D Point Cloud Construction

**Objective:**  
Generate an initial sparse 3D point cloud by triangulating matched keypoints.

**Details:**

- **Triangulation:**  
  Matched keypoints are triangulated using the two projection matrices.
  ```python
  def triangulation(proj_ref, proj_next, kp1, kp2):
      kp1_t = kp1.T  # (2, N)
      kp2_t = kp2.T  # (2, N)
      cloud = cv2.triangulatePoints(proj_ref, proj_next, kp1_t, kp2_t)
      cloud /= cloud[3]
      return kp1, kp2, cloud
  ```
- **Reprojection Error Calculation:**  
  The computed 3D points are reprojected back onto the image plane. The difference between the reprojected points and the original keypoints is used to calculate the reprojection error.
  ```python
  def reprojection_error(cloud, kp, trans_mat, K):
      R, _ = cv2.Rodrigues(trans_mat[:3, :3])
      kp_reprojected, _ = cv2.projectPoints(cloud, R, trans_mat[:3, 3], K, distCoeffs=None)
      kp_reprojected = np.float32(kp_reprojected[:, 0, :])
      error = cv2.norm(kp_reprojected, kp, cv2.NORM_L2)
      return error / len(kp_reprojected), cloud, kp_reprojected
  ```
- **Significance:**  
  A lower reprojection error indicates better accuracy of the triangulated 3D points.

---

## 7. Iterative PnP Registration and Point Cloud Update

**Objective:**  
Incrementally refine the 3D reconstruction by processing additional images.

**Details:**

- **Common Keypoint Matching Across Views:**  
  For each new image, the code identifies keypoints that appear in the previous images to maintain consistency.
  ```python
  def three_view_points(kp2, kp2_dash, kp3):
      # Identifies common and non-common keypoints across three consecutive images.
      ...
  ```
- **Pose Estimation via PnP:**  
  The Perspective-n-Point (PnP) algorithm is used to estimate the camera pose for the new image based on 3D-2D correspondences.
  ```python
  R, t, kp3_common, cloud_common, kp2_dash_common = perspective_n_point(cloud[index], kp3_common, kp2_dash_common, K)
  trans_mat_new = np.hstack((R, t))
  proj_new = K @ trans_mat_new
  ```
- **Triangulation of New Points:**  
  Both the common keypoints (for refining existing 3D points) and new keypoints (for expanding the cloud) are triangulated.
  ```python
  kp2_dash_uncommon, kp3_uncommon, cloud_new = triangulation(proj_2, proj_new, kp2_dash_uncommon, kp3_uncommon)
  ```
- **Updating the Reconstruction:**  
  The new 3D points are appended to the cumulative point cloud. The corresponding pixel colors are also stored for visualization.
- **Reprojection Error:**  
  Reprojection error is computed for the new points to validate the pose estimation and triangulation.


### Reprojection Error - Overview

After reconstructing a set of 3D points from multiple images, we want to verify that these points are consistent with the original image measurements. The reprojection error does exactly that by comparing:
- **Observed Keypoints:** The 2D feature points detected in the image.
- **Reprojected Points:** The 3D points projected back onto the image plane using the camera's intrinsic parameters and the estimated pose.

A lower reprojection error indicates that the 3D reconstruction and camera pose estimation are accurate.

#### Detailed Steps

#### 1. Extract the Camera Pose

The camera pose is represented by a transformation matrix that combines a rotation matrix \( R \) and a translation vector \( t \). These parameters are essential for projecting 3D points into the 2D image space.

- **Rotation Extraction:**  
  The rotation part is stored in the top-left 3×3 submatrix of the transformation matrix. To use it with OpenCV’s projection functions, it is converted using the Rodrigues transformation:
  ```python
  R, _ = cv2.Rodrigues(trans_mat[:3, :3])
  ```
  This converts the rotation matrix (or rotation vector) into a suitable format for projection.

#### 2. Project 3D Points to the Image Plane

With the camera pose \( (R, t) \) and the intrinsic camera matrix \( K \), we can project each 3D point (from the triangulated cloud) into the 2D image plane.

- **Using OpenCV’s `projectPoints`:**  
  The function `cv2.projectPoints` is used to compute the 2D image coordinates for each 3D point:
  ```python
  kp_reprojected, _ = cv2.projectPoints(cloud, R, trans_mat[:3, 3], K, distCoeffs=None)
  kp_reprojected = np.float32(kp_reprojected[:, 0, :])  # Reshape to (N, 2)
  ```
  Here, `cloud` is an array of 3D points (shape: \( N \times 3 \)), and the result `kp_reprojected` is an array of projected 2D points (shape: \( N \times 2 \)).

#### 3. Compute the Reprojection Error

The reprojection error is computed as the Euclidean (L2) distance between the reprojected 2D points and the original observed keypoints.

- **Error Calculation:**  
  For each point \( i \), the error is:
  \[
  \text{error}_i = \| \text{kp\_reprojected}_i - \text{kp}_i \|
  \]
  The overall error is typically averaged over all \( N \) points:
  ```python
  error = cv2.norm(kp_reprojected, kp, cv2.NORM_L2)
  mean_error = error / len(kp_reprojected)
  ```
  This mean reprojection error gives a single scalar value representing how well the 3D reconstruction fits the 2D observations.

#### 4. Code Reference

The following is the implementation of the `reprojection_error` function from the code:

```python
def reprojection_error(cloud, kp, trans_mat, K):
    """
    Projects 3D points back onto the image plane and computes the reprojection error.
    
    Parameters:
      - cloud: Array of 3D points (shape: (N, 3)).
      - kp: Observed 2D keypoints in the image (shape: (N, 2)).
      - trans_mat: Camera transformation matrix (3x4), containing rotation and translation.
      - K: Intrinsic camera matrix.
    
    Returns:
      - mean error: Average reprojection error over all keypoints.
      - cloud: The original 3D points.
      - kp_reprojected: The 2D projections of the 3D points.
    """
    try:
        # Extract the rotation from the transformation matrix.
        R, _ = cv2.Rodrigues(trans_mat[:3, :3])
        
        # Project the 3D points to the 2D image plane.
        kp_reprojected, _ = cv2.projectPoints(cloud, R, trans_mat[:3, 3], K, distCoeffs=None)
        kp_reprojected = np.float32(kp_reprojected[:, 0, :])
        
        # Ensure the shapes of reprojected and observed keypoints match.
        if kp_reprojected.shape != kp.shape:
            raise Exception(f"Shape mismatch: reprojected {kp_reprojected.shape} vs original {kp.shape}")
        
        # Calculate the total L2 error and compute the average.
        error = cv2.norm(kp_reprojected, kp, cv2.NORM_L2)
        return error / len(kp_reprojected), cloud, kp_reprojected
    except Exception as e:
        raise Exception("Error in reprojection_error: " + str(e))
```

#### 5. Interpretation of the Error

- **Low Error:**  
  Indicates that the projected 3D points closely match the observed keypoints, suggesting that the camera pose estimation and 3D reconstruction are accurate.
  
- **High Error:**  
  Suggests potential issues in the SfM pipeline, such as:
  - Incorrect feature matching.
  - Poor estimation of the camera pose.
  - Inaccurate triangulation of the 3D points.

The reprojection error is often used in **bundle adjustment**, a refinement step that optimizes both the 3D point positions and camera parameters by minimizing this error.

## 8. Final Output and Visualization

**Objective:**  
Export the reconstructed 3D point cloud into a PLY file for visualization and further analysis (e.g., in MeshLab).

**Details:**

- **Point Cloud Aggregation:**  
  All accumulated 3D points and their associated colors are combined.
- **Outlier Filtering:**  
  An optional cleaning step removes points that are far from the cloud's mean to reduce noise.
- **PLY Export:**  
  The final point cloud is saved in the PLY format.
  ```python
  def output(final_cloud, pixel_colour, output_filename):
      output_points = final_cloud.reshape(-1, 3) * 200  # Scaling factor
      output_colors = pixel_colour.reshape(-1, 3)
      mesh = np.hstack([output_points, output_colors])
      ...
      with open(output_filename, 'w') as f:
          f.write(ply_header % dict(vert_num=len(mesh)))
          np.savetxt(f, mesh, fmt='%f %f %f %d %d %d')
  ```

---

## 9. Error Handling and Robustness

**Objective:**  
Provide robust error handling and detailed logging at each stage of the pipeline to facilitate debugging and ensure stability.

**Details:**

- **Try/Except Blocks:**  
  Each major function (e.g., `subsample`, `triangulation`, `get_features`, `reprojection_error`) is wrapped in try/except blocks. This ensures that errors are caught and reported with context, such as:
  ```python
  try:
      kp1, kp2, cloud = triangulation(proj_1, proj_2, kp1, kp2)
  except Exception as e:
      print("Error during initial triangulation and PnP:", e)
      sys.exit(1)
  ```
- **Shape Verification:**  
  In the `reprojection_error` function, there is an explicit shape check to ensure that the projected and original keypoints have the same dimensions.
- **Command-line Interface:**  
  The use of `argparse` makes the code easy to run with different parameters (e.g., input folder, downsampling factor, output file) directly from the terminal.

---

