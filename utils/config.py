import numpy as np
import cv2


def get_norm(desc_name):
    """Returns the appropriate norm for the descriptor type."""
    if desc_name in ['SIFT', 'SURF']:
        return cv2.NORM_L2
    # ORB, BRISK, AKAZE use Hamming
    return cv2.NORM_HAMMING

def init_pipeline(det_name='GFTT', desc_name='ORB', matcher_name='FLANN', config = {}):
    """
    Initializes Detector, Descriptor, and Matcher based on dictionary configs.
    
    Args:
        det_name (str): 'GFTT', 'ORB', 'BRISK', 'AKAZE', 'FAST'
        desc_name (str): 'ORB', 'SIFT', 'BRISK', 'AKAZE'
        matcher_name (str): 'FLANN' or 'BF'
        
    Returns:
        detector, descriptor, matcher
    """

    # Initialize Detector
    detector_configs = config['detector']
    if det_name in detector_configs:
        cfg = detector_configs[det_name]
        detector = cfg['factory'](**cfg['params'])
    else:
        raise ValueError(f"Unknown detector: {det_name}")

    # Initialize Descriptor
    descriptor_config = config['descriptor']
    if desc_name in descriptor_config:
        cfg = descriptor_config[desc_name]
        descriptor = cfg['factory'](**cfg['params'])
    else:
        raise ValueError(f"Unknown descriptor: {desc_name}")

    # --- 3. Matcher Configuration ---
    norm = get_norm(desc_name)
    
    # Prepare Index Params based on Norm (Binary vs Float)
    if norm == cv2.NORM_HAMMING:
        # LSH parameters for Binary Descriptors (ORB, BRISK, AKAZE)
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    else:
        # KD-Tree parameters for Float Descriptors (SIFT)
        index_params = dict(algorithm=1, trees=5)
        
    search_params = dict(checks=50)

    if matcher_name == 'BF':
        # Brute Force Matcher
        # Note: BFMatcher does not accept index_params/search_params
        matcher = cv2.BFMatcher(norm, crossCheck=True)
        
    elif matcher_name == 'FLANN':
        # FLANN Matcher
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
    else:
        raise ValueError(f"Unknown matcher: {matcher_name}")

    return detector, descriptor, matcher