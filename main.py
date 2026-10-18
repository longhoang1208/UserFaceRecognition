
import cv2
import numpy as np
import mediapipe as mp
from drawing_utils import draw_bbox
from drawing_utils import draw_side_bar


# =====================================
# CONFIGURATIONS
# =====================================

# -------------------------------
# CAMERA READER
# -------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# -------------------------------
# MEDIAPIPE SOLUTIONS
# -------------------------------
mp_drawing = mp.solutions.drawing_utils

mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.7
)


# -------------------------------
# COLLECT ADMIN DATA
# -------------------------------
frame_data = []
num_frames = 300
counted_frames = 0
is_collecting = False


# -------------------------------
# ADMIN VERIFICATION
# -------------------------------
admin_data = None
admin_status = "Not found"
current_user = "Unknown"

# =====================================

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    result = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h, w = frame.shape[:2]
    
    if result.detections:
        for detection in result.detections:
            bbox = detection.location_data.relative_bounding_box

            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))

            x2 = min(w, int(x1 + bbox.width * w))
            y2 = min(h, int(y1 + bbox.height * h))

            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, (640, 640))

            draw_bbox(frame, x1, y1, x2, y2)

    if admin_data is not None:
        admin_status = "Active"
        difference = np.mean(
            np.abs(
                roi.astype(np.float32) - admin_data
            )
        )
        if difference < 25:
            current_user = "Admin"
        else:
            current_user = "Unknown"

        admin_img = cv2.resize(admin_data.copy(), (200, 200))

    else:
        admin_img = np.zeros((200, 200, 3), dtype=np.uint8)

    draw_side_bar(
        frame,
        admin_status,
        current_user,
        admin_img,
        admin_data
    )

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        is_collecting = True

    if is_collecting and roi is not None and counted_frames < num_frames:
        frame_data.append(roi)
        counted_frames += 1

        cv2.putText(
            frame,
            f"Scanning {int((counted_frames/num_frames)*100)}%...",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (150, 150, 150), 1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Please keep your face still.",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (150, 150, 150), 1,
            cv2.LINE_AA
        )

    elif is_collecting and counted_frames >= num_frames:
        if admin_data is None:
            admin_data = np.mean(np.stack(frame_data), axis=0)
        is_collecting = False
        frame_data.clear()

    if key == ord('r') or key == ord('R'):
        admin_data = None
        current_user = "Unknown"
        counted_frames = 0

    cv2.imshow("frame", frame)

    if key == 27:
        frame_data.clear()
        break

cap.release()
cv2.destroyAllWindows()