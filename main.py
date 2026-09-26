
import cv2
import numpy as np
import mediapipe as mp


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


mp_drawing = mp.solutions.drawing_utils

mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.7
)


frame_data = []
num_frames = 150
counted_frames = 0

is_collecting = False
admin_data = None

admin_status = "Not found"
current_user = "Unknown"


def draw_bbox(x1, y1, x2, y2):
    # TOP - RIGHT
    cv2.line(
        frame,
        (x1, y1),
        (x1, y1 + int((y2 - y1)/4)),
        (0, 255, 0), 2
    )

    cv2.line(
        frame,
        (x1, y1),
        (x1 + int((x2 - x1)/4), y1),
        (0, 255, 0), 2
    )

    # TOP - LEFT
    cv2.line(
        frame,
        (x2, y1),
        (x2, y1 + int((y2 - y1)/4)),
        (0, 255, 0), 2
    )

    cv2.line(
        frame,
        (x2, y1),
        (x2 - int((x2 - x1)/4), y1),
        (0, 255, 0), 2
    )

    # BOTTOM - RIGHT
    cv2.line(
        frame,
        (x1, y2),
        (x1, y2 - int((y2 - y1)/4)),
        (0, 255, 0), 2
    )

    cv2.line(
        frame,
        (x1, y2),
        (x1 + int((x2 - x1)/4), y2),
        (0, 255, 0), 2
    )

    # BOTTOM - LEFT
    cv2.line(
        frame,
        (x2, y2),
        (x2, y2 - int((y2 - y1)/4)),
        (0, 255, 0), 2
    )

    cv2.line(
        frame,
        (x2, y2),
        (x2 - int((x2 - x1)/4), y2),
        (0, 255, 0), 2
    )

    # DRAW CROSS
    cv2.line(
        frame,
        (x1 + int((x2 - x1)/2), y1 + int((y2 - y1)/1.8)),
        (x1 + int((x2 - x1)/2), y2 - int((y2 - y1)/1.8)),
        (0, 255, 0), 2
    )

    cv2.line(
        frame,
        (x1 + int((x2 - x1)/1.8), y1 + int((y2 - y1)/2)),
        (x2 - int((x2 - x1)/1.8), y1 + int((y2 - y1)/2)),
        (0, 255, 0), 2
    )


def draw_side_bar(frame: np.ndarray):
    h, w = frame.shape[:2]
    x1 = 10
    y1 = 20
    x2 = int(w/5)
    y2 = h - 100
    r = 10

    cv2.circle(
        frame,
        (x1 + r, y1 + r),
        r, (40, 40, 40), -1
    )

    cv2.circle(
        frame,
        (x2 - r, y1 + r),
        r, (40, 40, 40), -1
    )

    cv2.circle(
        frame,
        (x1 + r, y2 - r),
        r, (40, 40, 40), -1
    )

    cv2.circle(
        frame,
        (x2 - r, y2 - r),
        r, (40, 40, 40), -1,
        cv2.LINE_AA
    )

    cv2.rectangle(
        frame,
        (x1, y1 + r),
        (x1 + r, y2 - r),
        (40, 40, 40), -1,
        cv2.LINE_AA
    )

    cv2.rectangle(
        frame,
        (x2, y1 + r),
        (x2 - r, y2 - r),
        (40, 40, 40), -1,
        cv2.LINE_AA
    )

    cv2.rectangle(
        frame,
        (x1 + r, y1),
        (x2 - r, y2),
        (40, 40, 40), -1,
        cv2.LINE_AA
    )


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

            draw_bbox(x1, y1, x2, y2)

    draw_side_bar(frame)

    cv2.putText(
        frame,
        f"Admin: {admin_status}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0) if admin_data is not None
        else (0, 0, 255), 1, cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"User: {current_user}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0) if current_user != "Unknown"
        else (0, 0, 255), 1,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (20, 130),
        (int(w/5) - 10, 130),
        (0, 200, 240), 1
    )

    if admin_data is not None:
        admin_status = "Active"
        difference = np.mean(
            np.abs(
                roi.astype(np.float32) - admin_data
            )
        )
        if difference < 20:
            current_user = "Admin"
        else:
            current_user = "Unknown"

        admin_img = cv2.resize(admin_data.copy(), (200, 200))

    else:
        admin_img = np.zeros((200, 200, 3), dtype=np.uint8)

    cv2.putText(
        frame,
        "Current admin",
        (20, h - 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 1,
        cv2.LINE_AA
    )
    frame[h - 450:h - 450 + 200:, 30:30 + 200] = admin_img

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

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        is_collecting = True

    if is_collecting and counted_frames < num_frames:
        frame_data.append(roi)
        counted_frames += 1

        cv2.putText(
            frame,
            f"Scanning {counted_frames}/{num_frames}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (150, 150, 150), 1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Slowly turn your face left and right",
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

    if key == ord('r'):
        admin_data = None
        counted_frames = 0

    cv2.imshow("frame", frame)

    if key == 27:
        frame_data.clear()
        break

cap.release()
cv2.destroyAllWindows()