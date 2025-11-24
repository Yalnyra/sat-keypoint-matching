import cv2
import numpy as np

### Author: Eugen Vinokur 24/11/2025
### Clustering-discouraging keypoint detections methods
### Using sliding window and non-max suppression

def detect_patch(detector, img_patch, offset_x, offset_y, max_pts):
    """
    Runs detection on a specific image patch and shifts coordinates 
    back to global image space.
    """
    kps = detector.detect(img_patch, None)
    
    # Sort by response strength (strongest first)
    kps = sorted(kps, key=lambda x: x.response, reverse=True)
    
    # Cap the number of points (Prevention of "Clumping")
    kps = kps[:max_pts]
    
    # Adjust coordinates to global space
    for kp in kps:
        kp.pt = (kp.pt[0] + offset_x, kp.pt[1] + offset_y)
        
    return kps

def detect_window(detector, img_gray, patch_size=(4,4), max_pts=200, min_valid=0.1):
    """
    Detect fixed window
    
    """
    h, w = img_gray.shape[:2]
    patch_h, patch_w = patch_size
    
    # How many full patches fit?
    n_rows = h // patch_h
    n_cols = w // patch_w
    
    # Calculate centered starting points (splitting the residual)
    # If 50px remain, we start at index 25.
    y_margin = (h - (n_rows * patch_h)) // 2
    x_margin = (w - (n_cols * patch_w)) // 2
    
    all_kps = []

    # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)  
    # --- ITERATE OVER GRID ---
    for r in range(n_rows):
        for c in range(n_cols):
            # Calculate coordinates with margin offset
            y_start = y_margin + (r * patch_h)
            y_end = y_start + patch_h
            
            x_start = x_margin + (c * patch_w)
            x_end = x_start + patch_w
            # Extract the specific patch from the specific channel
            patch = img_gray[y_start:y_end, x_start:x_end]
            
            # Calculate ratio of valid (non-zero) pixels
            # This assumes mask is binary (0 for invalid, >0 for valid)
            valid_pixels = np.count_nonzero(patch)
            total_pixels = patch_h * patch_w
            
            if (valid_pixels / total_pixels) < min_valid:
                continue # Skip this patch
                    
            
            
            # Detect (pass absolute coordinates x_start/y_start for global mapping)
            patch_kps = detect_patch(detector, patch, x_start, y_start, max_pts)
            all_kps.extend(patch_kps)
    return all_kps

def detect_grid(detector, img_gray, grid_size=(4,4), max_pts=200, min_valid=0.1):
        """
        1. Splits image into RGB channels.
        2. Splits each channel into grid tiles.
        3. Detects on every tile of every channel.
        4. Fuses results.
        """
        all_kps = []
        grid_rows, grid_cols = grid_size
        h, w = img_gray.shape[:2]
        
        # Calculate cell dimensions
        step_h = h // grid_rows
        step_w = w // grid_cols
        
        # --- ITERATE OVER GRID ---
        for r in range(grid_rows):
            for c in range(grid_cols):
                # Define patch coordinates
                y_start, y_end = r * step_h, (r + 1) * step_h
                x_start, x_end = c * step_w, (c + 1) * step_w

                patch = img_gray[y_start:y_end, x_start:x_end]
                # --- VALIDITY CHECK ---
                # Calculate ratio of valid (non-zero) pixels
                # This assumes mask is binary (0 for invalid, >0 for valid)
                valid_pixels = np.count_nonzero(patch)
                total_pixels = step_h * step_w
                
                if (valid_pixels / total_pixels) < min_valid:
                    continue # Skip this patch

                
                
                # Detect (pass absolute coordinates x_start/y_start for global mapping)
                patch_kps = detect_patch(detector, patch, x_start, y_start, max_pts)
                all_kps.extend(patch_kps)

        return all_kps


def get_norm(desc_name):
    # Binary descriptors need HAMMING, Floats need L2
    if desc_name in ['ORB', 'BRISK', 'AKAZE', 'BRIEF']:
        return cv2.NORM_HAMMING
    return cv2.NORM_L2

def lowe_test(matches, nn_ratio=0.7):
    # store only the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < nn_ratio * n.distance:
            good.append(m)

    return good

def knn_match(desc_name, des1, des2):
  
    # FLANN parameters
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict(checks = 50)

    norm = get_norm(desc_name)
    # Configure Matcher
    if norm == cv2.NORM_HAMMING:
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    else:
        index_params = dict(algorithm=1, trees=5)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match features from each image
    matches = flann.knnMatch(des1, des2, k=2)

    if desc_name == 'SIFT':
        return lowe_test(matches, 0.7)
    else:
        return lowe_test(matches, 1.0)
    

def non_max_suppression(keypoints, max_points=50):
    """
    Implements simplified ANMS (Brown et al.).
    1. Sorts points by response (strength).
    2. For every point, finds the distance to the nearest *stronger* point.
    3. Selects points that have the largest distance to their stronger neighbors.
    """
    if len(keypoints) <= max_points:
        return keypoints

    # 1. Extract coordinates and responses
    pts = np.array([kp.pt for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])

    # 2. Sort by response (descending)
    # We want the strongest points first to serve as "anchors"
    idxs = np.argsort(responses)[::-1]
    pts = pts[idxs]
    original_indices = idxs # Keep track to return correct KeyPoint objects

    # Calculate suppression radius for each point
    n = len(pts)
    radii = np.full(n, np.inf)
    
    # Note: This is O(N^2) worst case. For production with >5000 points, 
    # use a KD-Tree (scipy.spatial.cKDTree).
    for i in range(1, n):
        # Compare point i with all stronger points (0 to i-1)
        # We add a small robust multiplier (0.9) to ensure we only respect 
        # neighbors that are strictly stronger/equal, handling floating point drift.
        
        # Calculate Euclidean distances to all stronger points
        dist_sq = np.sum((pts[:i] - pts[i])**2, axis=1)
        
        # The radius is the minimum distance to a stronger point
        radii[i] = np.min(dist_sq)

    # Select top k points with the largest suppression radii
    best_indices = np.argsort(radii)[::-1][:max_points]
    
    # Map back to original keypoint objects
    keep_indices = original_indices[best_indices]
    
    return [keypoints[i] for i in keep_indices]


### RGB descriptors for higher contrast 
def to_opponent_space(img_bgr):
    """
    Converts BGR image to Opponent Color Space (O1, O2, O3).
    Returns 3 separate UINT8 images safe for ORB/BRISK/AKAZE.
    """
    img_bgr = img_bgr.astype(np.float32)
    B, G, R = cv2.split(img_bgr)

    # --- Mathematical Conversion ---
    # O1 = (R - G) / sqrt(2)
    # O2 = (R + G - 2B) / sqrt(6)
    # O3 = (R + G + B) / sqrt(3)  <-- Intensity/Grayscale
    
    O1 = (R - G) / np.sqrt(2)
    O2 = (R + G - 2*B) / np.sqrt(6)
    O3 = (R + G + B) / np.sqrt(3)

    # --- Normalization to 0-255 (Uint8) ---
    # Necessary because ORB/BRISK cannot read negative floats
    # O1 range approx [-180, 180] -> Shift + Scale
    O1 = cv2.normalize(O1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    O2 = cv2.normalize(O2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    O3 = cv2.normalize(O3, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return [O1, O2, O3]


def opponent_desc(extractor, channels, kps):
    """
    Computes descriptors on all 3 channels and concatenates them.
    """
    # If no keypoints detected, return None
    if len(kps) == 0:
        return None

    # Compute individually
    # Note: We rely on the extractor's compute method
    _, des1 = extractor.compute(channels[0], kps)
    _, des2 = extractor.compute(channels[1], kps)
    _, des3 = extractor.compute(channels[2], kps)

    # Handle edge case where compute fails for some channels
    if des1 is None or des2 is None or des3 is None:
        return None

    return np.hstack([des1, des2, des3])
