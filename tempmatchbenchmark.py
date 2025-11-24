
import cv2
import numpy as np
import time
import pandas as pd
import argparse
import logging
import os 
import sys
import matplotlib.pyplot as plt
from utils.detection import detect_grid, detect_window, knn_match, non_max_suppression

# load global logger
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s')

# calculate the angle with the horizontal
def angle_horizontal(v):
    return -np.arctan2(v[1],v[0])

def knn_clasif(good_matches):
  best_template, highest_logprob = None, 0.0

  sum_good_matches = sum([len(gm) for gm in good_matches])
  for i, gm in enumerate(good_matches):
    logprob = len(gm)/sum_good_matches
    # save highest
    if logprob > highest_logprob:
      highest_logprob = logprob
      best_template = i
    logger.info('p(t_{} | x) = {:.4f}'.format(i, logprob))
  return best_template

def vis_figure(img, base, name, det_name, desc_name, args, patch_size=None, draw_grid=False):
    bn, ext = os.path.splitext(os.path.basename(base))
    
    # Handling input type variations (assuming img might be a class or array)
    if hasattr(img, 'shape'):
        h_graph, w_graph = img.shape[:2]
    else:
        # Fallback if 'img' is a custom object
        h_graph, w_graph = img.img.shape[:2]

    # Create figure
    plt.figure(figsize=(w_graph / args.dpi, h_graph / args.dpi), dpi=args.dpi)
    
    # Display the main image
    # Note: 'out' was undefined in your snippet, assuming 'img' or 'out' is the target
    image_to_show = img.img if hasattr(img, 'img') else img
    plt.imshow(image_to_show, 'gray')

    # --- GRID DRAWING LOGIC ---
    if draw_grid and patch_size is not None:
        ph, pw = patch_size
        
        # Re-calculate the grid logic to know where to draw lines
        n_rows = h_graph // ph
        n_cols = w_graph // pw
        
        y_margin = (h_graph - (n_rows * ph)) // 2
        x_margin = (w_graph - (n_cols * pw)) // 2
        
        # Calculate grid boundaries
        grid_top = y_margin
        grid_bottom = y_margin + (n_rows * ph)
        grid_left = x_margin
        grid_right = x_margin + (n_cols * pw)

        # Define Line Positions
        # x_lines: Start at left margin, step by patch width, up to right margin
        x_lines = [x_margin + (i * pw) for i in range(n_cols + 1)]
        # y_lines: Start at top margin, step by patch height, up to bottom margin
        y_lines = [y_margin + (i * ph) for i in range(n_rows + 1)]

        # Draw Vertical Lines (Red) - Constrained between top and bottom of grid
        plt.vlines(x=x_lines, ymin=grid_top, ymax=grid_bottom, colors='r', linewidth=1)

        # Draw Horizontal Lines (Red) - Constrained between left and right of grid
        plt.hlines(y=y_lines, xmin=grid_left, xmax=grid_right, colors='r', linewidth=1)

    # Save logic
    logger.info('Saving full image in {}/{}_fix{}'.format(args.output_path, bn, ext))
    plt.axis('off') # Optional: removes axis ticks for cleaner image
    plt.tight_layout(pad=0)
    plt.savefig('{}/{}_{}_{}_{}{}'.format(args.output_path, bn, name, det_name, desc_name, ext))
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification and Matching Using Local Features and Homography.')
    parser.add_argument('-t', dest='template_name', required=True, help='Template image')
    parser.add_argument('-q', dest='query_name', required=True, help='Query image')
    parser.add_argument('-o', dest='output_path', help='Output directory', default='.')
    parser.add_argument('-v', dest='verbosity', action='store_true', help='Increase output verbosity')
    parser.add_argument('-p', dest='photocopied', action='store_true', help='Use only if the image is scanned or photocopied, do not with photos!')
    parser.add_argument('--matches', dest='view_matches', action='store_true', help="Shows the matching result and the good matches")
    parser.add_argument('--tres', dest='treshold', type=int, help='Minimum good matches to pass the validation test')
    parser.add_argument('--grid', dest='patch_size', type=int, help='Width in pixels of the window in the detection grid')
    parser.add_argument('--pts', dest='max_pts', type=int, help='Maximum number of detection points in a single patch')
    parser.add_argument('--nms', dest='nms_pts', type=int, help='Maximum number of Non Max Suppression points')



    parser.set_defaults(view_matches=False)
    parser.set_defaults(photocopied=False)
    parser.set_defaults(treshold=10)
    parser.set_defaults(patch_size=240)
    parser.set_defaults(max_pts=400)
    parser.set_defaults(nms_pts=50)
    

    parser.set_defaults(dpi=96)

    args = parser.parse_args()

    detectors = {
            # 'SIFT': cv2.SIFT_create(nfeatures=2000),
            'ORB': cv2.ORB_create(nfeatures=2000),
            'BRISK': cv2.BRISK_create(),
            # 'AKAZE': cv2.AKAZE_create(),
            # 'FAST': cv2.FastFeatureDetector_create(threshold=20),
            'GFTT': cv2.GFTTDetector_create(maxCorners=2000) # Shi-Tomasi
        }

    # Note: FAST and GFTT are NOT descriptors, so they aren't in this list.
    descriptors = {
        'SIFT': cv2.SIFT_create(),
        'ORB': cv2.ORB_create(),
        'BRISK': cv2.BRISK_create(),
        # 'AKAZE': cv2.AKAZE_create()
    }

    results = []
    
    # Load Images
    query_img = cv2.imread(args.query_name, cv2.IMREAD_GRAYSCALE)
    template_img = cv2.imread(args.template_name, cv2.IMREAD_GRAYSCALE)
    
    if query_img is None or template_img is None:
        print("Error: Could not load images.")
        sys.exit()

    
    # Standard screen DPI (Dots Per Inch) is usually 96. 
    print(f"{'Detector':<10} | {'Descriptor':<10} | {'KP (Small)':<10} | {'Matches':<8} | {'Time (s)':<8} | {'Status'}")
    print("-" * 70)

    # --- LOOP THROUGH PERMUTATIONS ---
    for det_name, detector in detectors.items():
        for desc_name, extractor in descriptors.items():
            
            # Skip incompatible combinations
            # AKAZE Descriptor only works with AKAZE Keypoints (usually)
            if desc_name == 'AKAZE' and det_name != desc_name:
                continue
                
            try:
                start_time = time.time()
                
                # Detect Keypoints
                kp_small = detect_grid(detector, template_img, (args.patch_size,args.patch_size), args.max_pts)
                kp_big = detect_grid(detector, query_img, (args.patch_size,args.patch_size), args.max_pts)
                
                nms_kp_small = non_max_suppression(kp_small, args.nms_pts)
                nms_kp_big = non_max_suppression(kp_big, args.nms_pts)

                print(f"Total Template image Keypoints: {len(kp_small)} \n Non Max Suppression: {len(nms_kp_small)}")
                print(f"Total Query image Keypoints: {len(kp_big)} \n Non Max Suppression: {len(nms_kp_big)}")

                # Visualize detections
                kp_small_vis = cv2.drawKeypoints(template_img, nms_kp_small, None, color=(255,0,0), flags=0)
                kp_big_vis = cv2.drawKeypoints(query_img, nms_kp_big, None, color=(255,0,0), flags=0)
            
                vis_figure(kp_small_vis, args.template_name, 'detect_nms', det_name, desc_name, args)
                vis_figure(kp_big_vis, args.query_name, 'detect_nms', det_name, desc_name, args)

                # Compute Descriptors
                _, des_small = extractor.compute(template_img, nms_kp_small)
                _, des_big = extractor.compute(query_img, nms_kp_big)

                if des_small is None or des_big is None:
                    results.append([det_name, desc_name, len(nms_kp_small), 0, 0, "Fail: No Desc"])
                    continue

                # Match (Added non max suppression)
                good_matches = knn_match(desc_name, des_small, des_big)

                # Verify Homography (Did it actually work?)
                status = "Fail"
                src_pts = np.float32([nms_kp_small[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([nms_kp_big[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is None or len(good_matches) <= args.treshold:
                        
                    end_time = time.time()
                    duration = round(end_time - start_time, 4)
                    
                    print(f"{det_name:<10} | {desc_name:<10} | {len(nms_kp_small):<10} | {len(good_matches):<8} | {duration:<8} | {status}")
                    results.append([det_name, desc_name, len(nms_kp_small), len(good_matches), duration, status])
                    continue

                status = "Success"   
                matchesMask = mask.ravel().tolist()

                # Make it affine
                M[2,2] = 1.0
                M[2,0] = 0.0
                M[2,1] = 0.0

                # Calculate the rectangle enclosing the query image
                h,w = template_img.shape

                # Define the rectangle in the coordinates of the template image
                pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

                # transform the rectangle from the template "coordinates" to the query "coordinates"
                dst = cv2.perspectiveTransform(pts,M)

                if args.photocopied:
                    logger.info('Simplifying transformation matrix ...'.format(args.template_name))
                    # if the image is a photocopy or scanned we can assume that there is no shear in x or y.
                    # Thus we can simplify the transformation matrix M with only: rotation, scale and tranlation.

                    # calculate template "world" reference vectors
                    w_v = np.array([w-1,0])
                    h_v = np.array([h-1,0])

                    # calculate query "world" reference vectors
                    w_vp = (dst[3]-dst[0])[0]
                    h_vp = (dst[1]-dst[0])[0]

                    # We lost the angle and the scale given that scalation shares the same position in M with the shear transformation
                    # see https://upload.wikimedia.org/wikipedia/commons/2/2c/2D_affine_transformation_matrix.svg
                    
                    # estimate the angle using the top-horizontal line
                    angle = angle_horizontal(w_vp)
                    
                    # estimate the scale using the top-horizontal line and left-vertical line
                    scale_x = np.linalg.norm(w_vp) / np.linalg.norm(w_v)
                    scale_y = np.linalg.norm(h_vp) / np.linalg.norm(h_v)

                    # retrieve translation from original matrix M
                    M = np.matrix([[ scale_x * np.cos(angle) , np.sin(angle)           , M[0,2] ],
                                    [ -np.sin(angle)          , scale_y * np.cos(angle) , M[1,2] ],
                                    [ 0                       , 0                       , 1.     ]])

                    # retransform the rectangle with the new matrix
                    dst = cv2.perspectiveTransform(pts,M)
                
                # if bounding boxes are provided and we only have one template
                # crop those bounding boxes
                # using M^{-1} we go from query coordinates to template coordinates.
                img_templ_coords = cv2.warpPerspective(query_img, np.linalg.inv(M), (w,h))
                vis_figure(img_templ_coords, args.template_name, 'reproject', det_name, desc_name, args)

                if args.view_matches:
                    # draw the rectangle in the image
                    out = cv2.polylines(query_img,[np.int32(dst)],True,0,2, cv2.LINE_AA)
                    # show the matching features
                    params = dict(matchColor = (0,255,0), # draw matches in green color
                                    singlePointColor = None,
                                    matchesMask = matchesMask, # draw only inliers
                                    flags = 2)
                    ## draw the matches image 
                    out = cv2.drawMatches(template_img, nms_kp_small,
                                            query_img, nms_kp_big,
                                            good_matches, 
                                            None, **params)
                    
                    vis_figure(out, args.template_name, 'match', det_name, desc_name, args)
                
                end_time = time.time()
                duration = round(end_time - start_time, 4)
                
                print(f"{det_name:<10} | {desc_name:<10} | {len(nms_kp_small):<10} | {len(good_matches):<8} | {duration:<8} | {status}")
                results.append([det_name, desc_name, len(nms_kp_small), len(good_matches), duration, status])

            except Exception as e:
                print(f"{det_name} + {desc_name} Failed: {e}")
                    
    df = pd.DataFrame(results, columns=["Detector", "Descriptor", "KP_Count", "Good_Matches", "Time", "Status"])
    df = df.sort_values(by="Good_Matches", ascending=False)
    print("\nSummary Sorted by Matches:")
    print(df)
    df.to_csv('{}/report.csv'.format(args.output_path))