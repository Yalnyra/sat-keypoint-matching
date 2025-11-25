import cv2
import numpy as np
from scipy.spatial import KDTree

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

def lowe_test(matches, ratio):
    """
    Robust Lowe's Ratio Test that handles missing neighbors.
    """
    good_matches = []
    
    # Do not use "for m, n in matches" here. 
    # It assumes strictly 2 values and will crash on LSH results.
    for match in matches:
        # Check if we actually found 2 neighbors
        if len(match) == 2:
            m, n = match
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    return good_matches
    
def non_max_suppression(keypoints, max_points=50, robust_coeff=0.9):
    """
    Optimized ANMS using Scipy KD-Tree (Approximate).
    
    Complexity: O(N log N) build + O(N * K) query
    """
    n = len(keypoints)
    if n <= max_points:
        return keypoints

    # 1. Extract coordinates and responses
    pts = np.array([kp.pt for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])

    # 2. Sort by response (descending)
    # Strongest points come first (index 0 is strongest)
    idxs = np.argsort(responses)[::-1]
    pts = pts[idxs]
    responses = responses[idxs] # Needed for robust coefficient check
    original_indices = idxs 

    # 3. KD-Tree Optimization
    # We build the tree once on all points.
    tree = KDTree(pts)
    
    # We query K nearest neighbors for every point.
    # K=32 is a heuristic: if the nearest stronger neighbor is further than
    # the closest 32 points, this point is practically isolated enough.
    search_k = min(n, 32)
    
    # query returns (distances, neighbor_indices)
    # jobs=-1 uses all CPU cores
    dists, neighbor_idxs = tree.query(pts, k=search_k, workers=-1)

    # 4. Compute Suppression Radii
    radii = np.full(n, np.inf)

    # Iterate through the pre-computed neighbors to find the first "stronger" one
    # Note: Vectorizing this completely is hard because the "stop condition" 
    # (finding the first valid neighbor) varies per row. A fast loop is preferred.
    for i in range(1, n):
        # The neighbors are already sorted by distance by cKDTree
        row_indices = neighbor_idxs[i]
        row_dists = dists[i]
        
        for k in range(1, search_k): # Skip k=0 (self)
            neighbor_idx = row_indices[k]
            
            # CONDITION: Is the neighbor "stronger"?
            # Since we sorted 'pts' by response descending, a lower index 
            # usually means higher response. 
            # We explicitly check the robust coefficient logic: R_neighbor > R_i * 0.9
            if responses[neighbor_idx] > responses[i] * robust_coeff:
                radii[i] = row_dists[k]
                break 

    # 5. Select top k points with the largest suppression radii
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
