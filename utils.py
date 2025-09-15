import numpy as np
import cv2


def xywh_to_xyxy_denorm(box, image_size):
    """Converts bounding box from YOLO format to denormalized [x1, y1, x2, y2] format.

    Args:
        box (tuple): Normalized bounding box (x_center, y_center, width, height).
        image_size (tuple): Size of the image (width, height).

    Returns:
        tuple: Denormalized bounding box (x1, y1, x2, y2).
    """
    x_center, y_center, width, height = box
    x1 = int((x_center - width / 2) * image_size[0])
    x2 = int((x_center + width / 2) * image_size[0])
    y1 = int((y_center - height / 2) * image_size[1])
    y2 = int((y_center + height / 2) * image_size[1])
    return (x1, y1, x2, y2)


def bbox_union(boxes):
    """Returns a bounding box that covers all given boxes.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes [x1, y1, x2, y2].
    Returns:
        numpy.ndarray: Updated bounding box.
    """
    min_x = np.min(boxes[:, 0])
    min_y = np.min(boxes[:, 1])
    max_x = np.max(boxes[:, 2])
    max_y = np.max(boxes[:, 3])
    return np.array([min_x, min_y, max_x, max_y], np.int32)


def bbox_intersection(boxes, intersect_with_box):
    """Returns boxes that intersect with a given bounding box.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes with shape (N, 4).
        intersect_with_box (numpy.ndarray): Bounding box to check intersection with [x1, y1, x2, y2].
    Returns:
        numpy.ndarray: Boxes that intersect with the given box.
    """
    x1, y1, x2, y2 = intersect_with_box
    return boxes[
        (boxes[:, 0] >= x1)
        & (boxes[:, 1] >= y1)
        & (boxes[:, 2] <= x2)
        & (boxes[:, 3] <= y2)
    ]


def bbox_size(box):
    """Calculates the size of a bounding box.

    Args:
        box (numpy.ndarray): Bounding box coordinates [x1, y1, x2, y2].
    Returns:
        numpy.ndarray: Size of the bounding box [width, height].
    """
    return box[2:] - box[:2] if box is not None else 0


def add_margin(box, margin=(10, 10)):
    """Adds a margin to a bounding box.

    Args:
        box (numpy.ndarray): Bounding box coordinates [x1, y1, x2, y2].
        margin (tuple): Margin to add to each side.
    Returns:
        numpy.ndarray: Updated bounding box.
    """
    x1, y1, x2, y2 = box
    return np.array(
        [x1 - margin[0], y1 - margin[1], x2 + margin[0], y2 + margin[1]], np.int32
    )


def bbox_from_image_contours(image):
    """Extracts the bounding box of the largest contour in an image.

    Args:
        image (numpy.ndarray): Input image.
    Returns:
        numpy.ndarray: Bounding box coordinates [x1, y1, x2, y2].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu threshold: foreground (book/page/cover) -> white, background -> black
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Close small gaps along edges
    th = cv2.morphologyEx(
        th,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=2,
    )
    # Keep the largest blob (scanned book region)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, th
    c = max(cnts, key=cv2.contourArea)

    # Cover with a rectangle
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(int)
    x1, y1 = box[:, 0].min(), box[:, 1].min()
    x2, y2 = box[:, 0].max(), box[:, 1].max()

    # Ensure the box is within image bounds
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return np.array([x1, y1, x2, y2])
