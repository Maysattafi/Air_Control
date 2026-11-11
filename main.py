
import cv2
import mediapipe as mp
import pyautogui
import time
import math
from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

# Cursor smoothing: higher = smoother, lower = faster
SMOOTHING = 0.35

# Thumb-index distance required for clicking
PINCH_THRESHOLD = 0.045

# Delay between clicks
CLICK_COOLDOWN = 0.45

# Scrolling sensitivity
SCROLL_SENSITIVITY = 140

# Hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def is_scroll_gesture(hand):
    """
    Left hand:
    Index and middle fingers extended.
    Ring and pinky fingers folded.
    """

    index_extended = hand[8].y < hand[6].y
    middle_extended = hand[12].y < hand[10].y

    ring_folded = hand[16].y > hand[14].y
    pinky_folded = hand[20].y > hand[18].y

    return (
        index_extended
        and middle_extended
        and ring_folded
        and pinky_folded
    )


def draw_hand(frame, hand, label):

    height, width, _ = frame.shape
    points = []

    # Camera preview is mirrored.
    # The model receives the original, unmirrored frame.
    for landmark in hand:

        x = int((1.0 - landmark.x) * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Draw connections
    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    # Draw landmarks
    for point in points:

        cv2.circle(
            frame,
            point,
            5,
            (0, 0, 255),
            -1
        )

    # Display hand identity
    cv2.putText(
        frame,
        label,
        (points[0][0] + 10, points[0][1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 100, 0),
        2
    )

    return points


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    if not MODEL_PATH.exists():
        print("Hand detection model not found!")
        return

    # Configure MediaPipe
    options = mp.tasks.vision.HandLandmarkerOptions(

        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),

        running_mode=mp.tasks.vision.RunningMode.VIDEO,

        num_hands=2,

        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Open webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot access webcam!")
        return

    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()

    # Cursor variables
    smooth_x = None
    smooth_y = None

    # Click variables
    was_pinching = False
    last_click_time = 0

    # Scroll variables
    previous_scroll_y = None
    scroll_remainder = 0

    last_timestamp = 0

    print("--------------------------------")
    print("       AIRCONTROL STARTED")
    print("--------------------------------")
    print("RIGHT HAND:")
    print("- Point index finger: Move cursor")
    print("- Pinch thumb + index: Click")
    print()
    print("LEFT HAND:")
    print("- Raise index + middle fingers")
    print("- Move hand up/down: Scroll")
    print()
    print("Press Q to quit.")
    print("--------------------------------")

    try:

        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:

                success, original_frame = camera.read()

                if not success:
                    print("Cannot read camera frame.")
                    break

                # Mirror the display only.
                # Keep original_frame unmodified for MediaPipe.
                frame = cv2.flip(original_frame, 1)

                # Prepare image for MediaPipe
                rgb_frame = cv2.cvtColor(
                    original_frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # Timestamp for video detection
                timestamp = max(
                    last_timestamp + 1,
                    time.monotonic_ns() // 1_000_000
                )

                last_timestamp = timestamp

                # Detect hands
                results = landmarker.detect_for_video(
                    mp_image,
                    timestamp
                )

                right_hand_found = False
                scroll_gesture_found = False

                height, width, _ = frame.shape

                # Process each detected hand
                if results.hand_landmarks:

                    for i, hand in enumerate(results.hand_landmarks):

                        # Identify the hand
                        handedness = (
                            results.handedness[i][0].category_name
                        )

                        # Draw the hand
                        points = draw_hand(
                            frame,
                            hand,
                            handedness.upper() + " HAND"
                        )

                        # ==================================
                        # RIGHT HAND: MOUSE CONTROL
                        # ==================================

                        if handedness == "Right":

                            right_hand_found = True

                            index = hand[8]
                            thumb = hand[4]

                            # Index finger pointing
                            index_extended = (
                                hand[8].y < hand[6].y
                            )

                            if index_extended:

                                # Convert camera coordinates
                                # to screen coordinates.
                                margin = 0.08

                                x_ratio = (
                                    index.x - margin
                                ) / (1 - 2 * margin)

                                y_ratio = (
                                    index.y - margin
                                ) / (1 - 2 * margin)

                                target_x = int(
                                    (1 - clamp(x_ratio, 0, 1))
                                    * (screen_width - 1)
                                )

                                target_y = int(
                                    clamp(y_ratio, 0, 1)
                                    * (screen_height - 1)
                                )

                                # Keep cursor away from
                                # the emergency corner.
                                target_x = clamp(
                                    target_x, 20, screen_width - 20
                                )

                                target_y = clamp(
                                    target_y, 20, screen_height - 20
                                )

                                # Smooth cursor movement
                                if smooth_x is None:

                                    smooth_x = target_x
                                    smooth_y = target_y

                                else:

                                    smooth_x = (
                                        smooth_x * (1 - SMOOTHING)
                                        + target_x * SMOOTHING
                                    )

                                    smooth_y = (
                                        smooth_y * (1 - SMOOTHING)
                                        + target_y * SMOOTHING
                                    )

                                # Move actual mouse
                                pyautogui.moveTo(
                                    int(smooth_x),
                                    int(smooth_y)
                                )

                            # Draw cursor tracking point
                            cv2.circle(
                                frame,
                                points[8],
                                12,
                                (255, 100, 0),
                                2
                            )

                            # Pinch detection
                            pinch_distance = math.hypot(
                                thumb.x - index.x,
                                thumb.y - index.y
                            )

                            is_pinching = (
                                pinch_distance < PINCH_THRESHOLD
                            )

                            current_time = time.monotonic()

                            # Click only when pinch STARTS
                            if (
                                is_pinching
                                and not was_pinching
                                and current_time - last_click_time
                                > CLICK_COOLDOWN
                            ):

                                pyautogui.click()

                                last_click_time = current_time

                            was_pinching = is_pinching

                            # Display pinch status
                            if is_pinching:

                                cv2.putText(
                                    frame,
                                    "CLICK!",
                                    (points[8][0] + 15, points[8][1]),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 255),
                                    2
                                )

                        # ==================================
                        # LEFT HAND: SCROLL CONTROL
                        # ==================================

                        elif handedness == "Left":

                            if is_scroll_gesture(hand):

                                scroll_gesture_found = True

                                # Calculate average position
                                # of index and middle fingers.
                                current_scroll_y = (
                                    hand[8].y + hand[12].y
                                ) / 2

                                if previous_scroll_y is not None:

                                    movement = (
                                        previous_scroll_y
                                        - current_scroll_y
                                    )

                                    scroll_remainder += (
                                        movement * SCROLL_SENSITIVITY
                                    )

                                    scroll_amount = int(
                                        scroll_remainder
                                    )

                                    if scroll_amount != 0:

                                        pyautogui.scroll(
                                            scroll_amount
                                        )

                                        scroll_remainder -= (
                                            scroll_amount
                                        )

                                previous_scroll_y = current_scroll_y

                                cv2.putText(
                                    frame,
                                    "SCROLLING",
                                    (points[0][0] + 10,
                                     points[0][1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 255),
                                    2
                                )

                # Reset click state if right hand disappears
                if not right_hand_found:

                    was_pinching = False
                    smooth_x = None
                    smooth_y = None

                # Reset scroll tracking if scroll gesture disappears
                if not scroll_gesture_found:

                    previous_scroll_y = None
                    scroll_remainder = 0

                # ==================================
                # DISPLAY INSTRUCTIONS
                # ==================================

                cv2.putText(
                    frame,
                    "RIGHT: Move + Pinch Click",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "LEFT: Two-Finger Scroll",
                    (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Press Q to quit",
                    (15, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                # Show webcam
                cv2.imshow("AirControl", frame)

                # Exit with Q
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except pyautogui.FailSafeException:

        print("Emergency stop activated!")

    finally:

        camera.release()
        cv2.destroyAllWindows()

        print("AirControl stopped.")


if __name__ == "__main__":
    main()
