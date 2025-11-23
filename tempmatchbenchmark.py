
import cv2
import numpy as np
import time
import pandas as pd
import argparse
import logging
import os 
import sys
import matplotlib.pyplot as plt


# load global logger
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s')


def get_norm(desc_name):
    # Binary descriptors need HAMMING, Floats need L2
    if desc_name in ['ORB', 'BRISK', 'AKAZE', 'BRIEF']:
        return cv2.NORM_HAMMING
    return cv2.NORM_L2

def knn_match(des1, des2, nn_ratio=0.7):
  
  # FLANN parameters
  index_params = dict(algorithm = 0, trees = 5)
  search_params = dict(checks = 50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)
  
  # Match features from each image
  matches = flann.knnMatch(des1, des2, k=2)

  # store only the good matches as per Lowe's ratio test.
  good = []
  for m, n in matches:
    if m.distance < nn_ratio * n.distance:
      good.append(m)

  return good

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification and Matching Using Local Features and Homography.')
    parser.add_argument('-t', dest='template_name', required=True, help='Template image')
    parser.add_argument('-q', dest='query_name', required=True, help='Query image')
    parser.add_argument('-o', dest='output_path', help='Output directory', default='.')
    parser.add_argument('-v', dest='verbosity', action='store_true', help='Increase output verbosity')
    parser.add_argument('--matches', dest='view_matches', action='store_true', help="Shows the matching result and the good matches")

    parser.set_defaults(view_matches=False)
    parser.set_defaults(photocopied=False)

    args = parser.parse_args()

    detectors = {
            'SIFT': cv2.SIFT_create(nfeatures=2000),
            'ORB': cv2.ORB_create(nfeatures=2000),
            'BRISK': cv2.BRISK_create(),
            'AKAZE': cv2.AKAZE_create(),
            'FAST': cv2.FastFeatureDetector_create(threshold=20),
            'GFTT': cv2.GFTTDetector_create(maxCorners=2000) # Shi-Tomasi
        }

    # --- 2. Define Descriptors ---
    # Note: FAST and GFTT are NOT descriptors, so they aren't in this list.
    descriptors = {
        'SIFT': cv2.SIFT_create(),
        'ORB': cv2.ORB_create(),
        'BRISK': cv2.BRISK_create(),
        'AKAZE': cv2.AKAZE_create()
    }

    results = []
    
    # Load Images
    query_img = cv2.imread(args.query_name, cv2.IMREAD_GRAYSCALE)
    template_img = cv2.imread(args.template_name, cv2.IMREAD_GRAYSCALE)
    
    if query_img is None or template_img is None:
        print("Error: Could not load images.")
        sys.exit()

    print(f"{'Detector':<10} | {'Descriptor':<10} | {'KP (Small)':<10} | {'Matches':<8} | {'Time (s)':<8} | {'Status'}")
    print("-" * 70)

    # --- LOOP THROUGH PERMUTATIONS ---
    for det_name, detector in detectors.items():
        for desc_name, extractor in descriptors.items():
            
            # Skip incompatible combinations
            # AKAZE Descriptor only works with AKAZE Keypoints (usually)
            if desc_name == 'AKAZE' and det_name != 'AKAZE':
                continue
                
            try:
                start_time = time.time()
                
                # A. Detect Keypoints
                kp_small = detector.detect(template_img, None)
                kp_big = detector.detect(query_img, None)
                
                # B. Compute Descriptors
                # Note: We use the 'extractor' object here, not the detector
                _, des_small = extractor.compute(template_img, kp_small)
                _, des_big = extractor.compute(query_img, kp_big)

                if des_small is None or des_big is None:
                    results.append([det_name, desc_name, len(kp_small), 0, 0, "Fail: No Desc"])
                    continue

                # C. Match
                norm = get_norm(desc_name)
                
                # Configure Matcher
                if norm == cv2.NORM_HAMMING:
                    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
                else:
                    index_params = dict(algorithm=1, trees=5)
                    
                matcher = cv2.FlannBasedMatcher(index_params, dict(checks=50))
                matches = matcher.knnMatch(des_small, des_big, k=2)

                # D. Filter (Ratio Test)
                good_matches = []
                ratio = 0.75 if norm == cv2.NORM_HAMMING else 0.7
                for m, n in matches:
                    if m.distance < ratio * n.distance:
                        good_matches.append(m)

                # E. Verify Homography (Did it actually work?)
                status = "Fail"
                src_pts = np.float32([kp_small[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_big[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is None or len(good_matches) <= 10:
                        
                    end_time = time.time()
                    duration = round(end_time - start_time, 4)
                    
                    print(f"{det_name:<10} | {desc_name:<10} | {len(kp_small):<10} | {len(good_matches):<8} | {duration:<8} | {status}")
                    results.append([det_name, desc_name, len(kp_small), len(good_matches), duration, status])
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
                    logger.info('Simplifying transformation matrix ...'.format(template_img_path))
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
                bn, ext = os.path.splitext(os.path.basename(template_img_path))
                # using M^{-1} we go from query coordinates to template coordinates.
                img_templ_coords = cv2.warpPerspective(query_img, np.linalg.inv(M), (w,h))
                logger.info('Saving full image in {}/{}_fix{}'.format(args.output_path, bn, ext))
                cv2.imwrite('{}/{}_fix{}'.format(args.output_path, bn, ext), img_templ_coords)

                if args.view_matches:
                    # draw the rectangle in the image
                    out = cv2.polylines(query_img,[np.int32(dst)],True,0,2, cv2.LINE_AA)
                    # show the matching features
                    params = dict(matchColor = (0,255,0), # draw matches in green color
                                    singlePointColor = None,
                                    matchesMask = matchesMask, # draw only inliers
                                    flags = 2)
                    ## draw the matches image 
                    out = cv2.drawMatches(template_img, kp_small,
                                            query_img, kp_big,
                                            good_matches, 
                                            None, **params)
                    ## show result
                    # --- START EDIT ---
                    # Calculate the dimensions of the combined image (template + query)
                    h_graph, w_graph = out.shape[:2]
                    
                    # Standard screen DPI (Dots Per Inch) is usually 96. 
                    # We convert pixels to inches for Matplotlib: inches = pixels / dpi
                    dpi = 96 
                    
                    # Create a figure that matches the pixel resolution of the output image
                    plt.figure(figsize=(w_graph / dpi, h_graph / dpi), dpi=dpi)
                    # --- END EDIT ---
                    ## show result
                    plt.imshow(out, 'gray')
                    plt.savefig('{}/{}_match_{}_{}{}'.format(args.output_path, bn, det_name, desc_name, ext), img_templ_coords)
                    plt.show()
                
                end_time = time.time()
                duration = round(end_time - start_time, 4)
                
                print(f"{det_name:<10} | {desc_name:<10} | {len(kp_small):<10} | {len(good_matches):<8} | {duration:<8} | {status}")
                results.append([det_name, desc_name, len(kp_small), len(good_matches), duration, status])

            except Exception as e:
                print(f"{det_name} + {desc_name} Failed: {e}")
                    
    df = pd.DataFrame(results, columns=["Detector", "Descriptor", "KP_Count", "Good_Matches", "Time", "Status"])
    print("\nSummary Sorted by Matches:")
    df.sort_values(by="Good_Matches", ascending=False).to_csv('{}/report.csv'.format(args.output_path))