"""
Digital Whiteboard - write and draw in the air using your hand (webcam).

Gestures:
- Only INDEX finger up  -> draw (or hover-select a toolbar button)
- INDEX + MIDDLE up     -> pen up (move without drawing)
- Hold your fingertip over a toolbar button for ~0.7s to select it
  (a small progress ring fills up while you hover)

Toolbar (top bar, hover to select):
- 5 color swatches
- Eraser
- Brush size (cycles S -> M -> L -> XL)
- Undo
- Clear (wipes the board - now a deliberate button, not a gesture)
- Save (writes board_output.png)

Keyboard fallback (in case a hover-select feels finicky):
  1-5 color | e eraser | z undo | c clear | s save | q / ESC quit

Requirements:
    pip install opencv-python mediapipe numpy
Run:
    python digital_whiteboard.py
"""

import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# ---------------------------------------------------------------------------
# Hand tracking setup
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------------------------------------------------------------------------
# Whiteboard state
# ---------------------------------------------------------------------------
canvas = None
prev_point = None

PEN_COLORS = [
    (0, 0, 255),    # red   (BGR)
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 255, 255) # white
]
BRUSH_SIZES = [4, 8, 16, 28]          # S, M, L, XL
BRUSH_LABELS = ["S", "M", "L", "XL"]
ERASER_SIZE = 40

current_color_index = 0
current_brush_index = 1     # start at "M"
tool = "pen"                 # "pen" or "eraser"

history = deque(maxlen=15)   # canvas snapshots for undo
stroke_in_progress = False   # so we only push one history snapshot per stroke

HOVER_SELECT_SECONDS = 0.7
hover_target = None
hover_start_time = None

TOOLBAR_HEIGHT = 70


def build_toolbar(w):
    """Return list of button dicts laid out across the top of the frame."""
    buttons = []
    x = 10
    pad = 10
    size = TOOLBAR_HEIGHT - 20

    for i, color in enumerate(PEN_COLORS):
        buttons.append({
            "id": f"color_{i}",
            "type": "color",
            "index": i,
            "rect": (x, 10, x + size, 10 + size),
            "color": color,
        })
        x += size + pad

    x += 15
    buttons.append({"id": "eraser", "type": "eraser",
                     "rect": (x, 10, x + size + 30, 10 + size)})
    x += size + 30 + pad

    buttons.append({"id": "brush", "type": "brush",
                     "rect": (x, 10, x + size + 30, 10 + size)})
    x += size + 30 + pad

    buttons.append({"id": "undo", "type": "undo",
                     "rect": (x, 10, x + size + 30, 10 + size)})
    x += size + 30 + pad

    buttons.append({"id": "clear", "type": "clear",
                     "rect": (x, 10, x + size + 40, 10 + size)})
    x += size + 40 + pad

    buttons.append({"id": "save", "type": "save",
                     "rect": (x, 10, x + size + 30, 10 + size)})

    return buttons


def point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


def push_history():
    history.append(canvas.copy())


def apply_button_action(btn):
    global current_color_index, tool, current_brush_index, canvas
    if btn["type"] == "color":
        current_color_index = btn["index"]
        tool = "pen"
    elif btn["type"] == "eraser":
        tool = "eraser"
    elif btn["type"] == "brush":
        current_brush_index = (current_brush_index + 1) % len(BRUSH_SIZES)
    elif btn["type"] == "undo":
        if history:
            canvas = history.pop()
    elif btn["type"] == "clear":
        push_history()
        canvas[:] = 0
    elif btn["type"] == "save":
        cv2.imwrite("board_output.png", canvas)
        print("Saved board_output.png")


def draw_toolbar(frame, buttons):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], TOOLBAR_HEIGHT), (25, 25, 25), -1)
    for btn in buttons:
        x1, y1, x2, y2 = btn["rect"]
        if btn["type"] == "color":
            cv2.rectangle(frame, (x1, y1), (x2, y2), btn["color"], -1)
            if btn["index"] == current_color_index and tool == "pen":
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 2)
        elif btn["type"] == "eraser":
            active = tool == "eraser"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), -1)
            cv2.putText(frame, "ERASE", (x1 + 6, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255) if active else (180, 180, 180), 2)
            if active:
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 2)
        elif btn["type"] == "brush":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), -1)
            cv2.putText(frame, f"SZ:{BRUSH_LABELS[current_brush_index]}", (x1 + 6, y1 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        elif btn["type"] == "undo":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), -1)
            cv2.putText(frame, "UNDO", (x1 + 6, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
        elif btn["type"] == "clear":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 150), -1)
            cv2.putText(frame, "CLEAR", (x1 + 6, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
        elif btn["type"] == "save":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 120, 0), -1)
            cv2.putText(frame, "SAVE", (x1 + 6, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)


def draw_hover_progress(frame, btn, progress):
    # Ring indicator disabled — hover-select still works the same,
    # it just no longer draws the yellow progress ring while you wait.
    pass


def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    lm = hand_landmarks.landmark
    fingers = []
    fingers.append(1 if lm[tips_ids[0]].x < lm[tips_ids[0] - 1].x else 0)
    for i in range(1, 5):
        fingers.append(1 if lm[tips_ids[i]].y < lm[tips_ids[i] - 2].y else 0)
    return fingers


print("Digital Whiteboard started. Hover over toolbar buttons to select them.")
print("Keys: 1-5 color | e eraser | z undo | c clear | s save | q/ESC quit")

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    buttons = build_toolbar(w)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    status_text = "Show your hand to the camera"

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        fingers = fingers_up(hand_landmarks)
        index_tip = hand_landmarks.landmark[8]
        x, y = int(index_tip.x * w), int(index_tip.y * h)
        total_up = sum(fingers)

        in_toolbar = y <= TOOLBAR_HEIGHT

        if in_toolbar and fingers[1] == 1 and total_up == 1:
            # Hovering in the toolbar zone with index finger -> selection mode
            prev_point = None
            stroke_in_progress = False
            hovered_btn = None
            for btn in buttons:
                if point_in_rect(x, y, btn["rect"]):
                    hovered_btn = btn
                    break

            if hovered_btn is not None:
                if hover_target == hovered_btn["id"]:
                    elapsed = time.time() - hover_start_time
                    progress = min(elapsed / HOVER_SELECT_SECONDS, 1.0)
                    draw_hover_progress(frame, hovered_btn, progress)
                    if progress >= 1.0:
                        apply_button_action(hovered_btn)
                        hover_target = None
                        hover_start_time = None
                else:
                    hover_target = hovered_btn["id"]
                    hover_start_time = time.time()
                status_text = f"Hovering: {hovered_btn['id']}"
            else:
                hover_target = None
                hover_start_time = None

            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)

        elif fingers[1] == 1 and fingers[2] == 1 and total_up == 2:
            # Pen up - move without drawing
            prev_point = None
            stroke_in_progress = False
            cv2.circle(frame, (x, y), 10, (255, 255, 0), 2)
            status_text = "Pen up (repositioning)"

        elif fingers[1] == 1 and total_up == 1:
            # Draw / erase on the canvas
            draw_color = (0, 0, 0) if tool == "eraser" else PEN_COLORS[current_color_index]
            thickness = ERASER_SIZE if tool == "eraser" else BRUSH_SIZES[current_brush_index]

            if not stroke_in_progress:
                push_history()
                stroke_in_progress = True

            cv2.circle(frame, (x, y), max(thickness // 2, 6), draw_color if tool != "eraser" else (120, 120, 120), -1)
            if prev_point is not None:
                cv2.line(canvas, prev_point, (x, y), draw_color, thickness)
            prev_point = (x, y)
            status_text = f"{'Erasing' if tool == 'eraser' else 'Drawing'}..."

        else:
            prev_point = None
            stroke_in_progress = False
            hover_target = None
            hover_start_time = None
            status_text = "Raise only your index finger to draw"

    else:
        prev_point = None
        stroke_in_progress = False
        hover_target = None
        hover_start_time = None

    # Composite canvas over the live frame
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask_inv)
    combined = cv2.add(frame_bg, canvas_fg)

    draw_toolbar(combined, buttons)

    cv2.putText(combined, status_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    cv2.imshow("Digital Whiteboard", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('c'):
        push_history()
        canvas[:] = 0
    elif key == ord('s'):
        cv2.imwrite("board_output.png", canvas)
        print("Saved board_output.png")
    elif key == ord('e'):
        tool = "eraser"
    elif key == ord('z'):
        if history:
            canvas = history.pop()
    elif key in [ord(str(n)) for n in range(1, 6)]:
        current_color_index = int(chr(key)) - 1
        tool = "pen"

cap.release()
cv2.destroyAllWindows()