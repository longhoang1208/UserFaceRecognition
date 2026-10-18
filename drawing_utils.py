

import cv2
import numpy as np

class Colors:
    YELLOW = (0, 200, 240)
    GREEN  = (0, 255, 0)
    RED    = (0, 0, 255)
    WHITE  = (255, 255, 255)
    BLACK  = (40, 40, 40)

COL = Colors()


# -------------------------------
# ROI & BOUNDING BOX
# -------------------------------
roi = None
bbox_corner_size = 40
bbox_center_mid_size = 20


def draw_bbox(frame, x1, y1, x2, y2):
    center = int((x1 + x2)/2)
    middle = int((y1 + y2)/2)

    rect_thick = 2
    corner_thick = 8
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        COL.GREEN, rect_thick,
        cv2.LINE_AA
    )

    # TOP - RIGHT
    cv2.line(
        frame,
        (x1, y1),
        (x1, y1 + bbox_corner_size),
        COL.GREEN, corner_thick
    )

    cv2.line(
        frame,
        (x1, y1),
        (x1 + bbox_corner_size, y1),
        COL.GREEN, corner_thick
    )

    # TOP - LEFT
    cv2.line(
        frame,
        (x2, y1),
        (x2, y1 + bbox_corner_size),
        COL.GREEN, corner_thick
    )

    cv2.line(
        frame,
        (x2, y1),
        (x2 - bbox_corner_size, y1),
        COL.GREEN, corner_thick
    )

    # BOTTOM - RIGHT
    cv2.line(
        frame,
        (x1, y2),
        (x1, y2 - bbox_corner_size),
        COL.GREEN, corner_thick
    )

    cv2.line(
        frame,
        (x1, y2),
        (x1 + bbox_corner_size, y2),
        COL.GREEN, corner_thick
    )

    # BOTTOM - LEFT
    cv2.line(
        frame,
        (x2, y2),
        (x2, y2 - bbox_corner_size),
        COL.GREEN, corner_thick
    )

    cv2.line(
        frame,
        (x2, y2),
        (x2 - bbox_corner_size, y2),
        COL.GREEN, corner_thick
    )


    cv2.line(
        frame,
        (center, y1),
        (center, y1 + bbox_center_mid_size),
        COL.GREEN, rect_thick,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (center, y2),
        (center, y2 - bbox_center_mid_size),
        COL.GREEN, rect_thick,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (x1, middle),
        (x1 + bbox_center_mid_size, middle),
        COL.GREEN, rect_thick,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (x2, middle),
        (x2 - bbox_center_mid_size, middle),
        COL.GREEN, rect_thick,
        cv2.LINE_AA
    )


def draw_side_bar(
        frame: np.ndarray,
        admin_status: str,
        current_user: str,
        admin_img: np.ndarray,
        admin_data: np.ndarray
    ):
    h, w = frame.shape[:2]
    x1 = 10
    y1 = 20
    x2 = int(w/5)
    y2 = h - 100
    r = 10

    """
    -------------------------------
    DRAW BOX
    -------------------------------
    """
    # TOP-LEFT CORNER
    cv2.circle(
        frame,
        (x1 + r, y1 + r),
        r, COL.BLACK, -1,
        cv2.LINE_AA
    )

    # TOP-RIGGHT CORNER
    cv2.circle(
        frame,
        (x2 - r, y1 + r),
        r, COL.BLACK, -1,
        cv2.LINE_AA
    )

    # BOTTOM-LEFT CORNER
    cv2.circle(
        frame,
        (x1 + r, y2 - r),
        r, COL.BLACK, -1,
        cv2.LINE_AA
    )

    # BOTTOM-RIGHT CORNER
    cv2.circle(
        frame,
        (x2 - r, y2 - r),
        r, COL.BLACK, -1,
        cv2.LINE_AA
    )

    # TOP-LEFT CORNER BORDER
    cv2.circle(
        frame,
        (x1 + r, y1 + r),
        r, COL.YELLOW, 1,
        cv2.LINE_AA
    )

    # TOP-RIGHT CORNER BORDER
    cv2.circle(
        frame,
        (x2 - r, y1 + r),
        r, COL.YELLOW, 1,
        cv2.LINE_AA
    )

    # BOTTOM-LEFT CORNER BORDER
    cv2.circle(
        frame,
        (x1 + r, y2 - r),
        r, COL.YELLOW, 1,
        cv2.LINE_AA
    )

    # BOTTOM-RIGHT CORNER BORDER
    cv2.circle(
        frame,
        (x2 - r, y2 - r),
        r, COL.YELLOW, 1,
        cv2.LINE_AA
    )

    # FILL BOX SIDE LEFT
    cv2.rectangle(
        frame,
        (x1, y1 + r),
        (x1 + r, y2 - r),
        COL.BLACK, -1,
        cv2.LINE_AA
    )

    # FILL BOX SIDE RIGHT
    cv2.rectangle(
        frame,
        (x2, y1 + r),
        (x2 - r, y2 - r),
        COL.BLACK, -1,
        cv2.LINE_AA
    )

    # FILL INNER BOX
    cv2.rectangle(
        frame,
        (x1 + r, y1),
        (x2 - r, y2),
        COL.BLACK, -1,
        cv2.LINE_AA
    )

    # LEFT BORDER
    cv2.line(
        frame,
        (x1, y1 + r),
        (x1, y2 - r),
        COL.YELLOW, 1,
        cv2.LINE_AA
    )

    # RIGHT BORDER
    cv2.line(
        frame,
        (x2, y1 + r),
        (x2, y2 - r),
        COL.YELLOW, 1,
        cv2.LINE_AA
    )

    # TOP BORDER
    cv2.line(
        frame,
        (x1 + r, y1),
        (x2 - r, y1),
        COL.YELLOW, 1,
        cv2.LINE_AA
    )

    # BOTTOM BORDER
    cv2.line(
        frame,
        (x1 + r, y2),
        (x2 - r, y2),
        COL.YELLOW, 1,
        cv2.LINE_AA
    )

    # DRAW HORZ LINE
    cv2.line(
        frame,
        (x1, 130),
        (x2, 130),
        COL.YELLOW, 1
    )


    """
    -------------------------------
    TEXTS
    -------------------------------
    """
    # ADMIN STATUS
    cv2.putText(
        frame,
        f"Admin: {admin_status}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COL.GREEN if admin_data is not None
        else COL.RED, 1, cv2.LINE_AA
    )

    # USER DETECTED
    cv2.putText(
        frame,
        f"User: {current_user}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COL.GREEN if current_user != "Unknown"
        else COL.RED, 1,
        cv2.LINE_AA
    )

    # CURRENT ADMIN IMAGE
    cv2.putText(
        frame,
        "Current admin",
        (20, h - 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, COL.WHITE, 1,
        cv2.LINE_AA
    )
    # Replace the area by admin img
    frame[h - 450:h - 450 + 200:, 30:30 + 200] = admin_img

    # USER GUIDES
    cv2.putText(
        frame,
        "Press 'R' to remove current admin",
        (20, h - 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4, (180, 180, 180), 1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Press SPACE to scan new admin",
        (20, h - 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4, (180, 180, 180), 1,
        cv2.LINE_AA
    )