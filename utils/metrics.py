def box_iou(boxA, boxB):
    """
    Computes IoU between two rectangles.
    Boxes are formatted as: [x, y, w, h]
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection rectangle
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute the intersection over union
    # Union = AreaA + AreaB - Intersection
    unionArea = float(boxAArea + boxBArea - interArea)
    
    if unionArea == 0:
        return 0.0

    iou = interArea / unionArea
    return iou