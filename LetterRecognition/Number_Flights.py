import cv2
import numpy as np
import threading
import time
from djitellopy import Tello
import tensorflow as tf
import av

# ========================
# Drone Movement Functions
# ========================

def diagonal_up_right(tello, distance=100, speed=30):
    tello.send_rc_control(speed, 0, speed, 0)
    time.sleep(distance / speed)
    tello.send_rc_control(0, 0, 0, 0)

def curve_up_right(tello, distance, speed):
    diagonal_up_right(tello, distance, speed)
    diagonal_up_right(tello, distance, int(speed*1.5))
    tello.send_rc_control(0, 0, speed, 0)
    time.sleep(distance / speed / 3)
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(distance / speed / 3)

def curve_up_left(tello, distance, speed):
    tello.send_rc_control(0, 0, speed, 0)
    time.sleep(distance / speed / 3)
    tello.send_rc_control(-speed, 0, int(speed*1.5), 0)
    time.sleep(distance / speed / 3)
    tello.send_rc_control(-speed, 0, speed, 0)
    time.sleep(distance / speed / 3)
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(distance / speed / 3)

def curve_down_left(tello, distance, speed):
    tello.send_rc_control(-speed, 0, -speed, 0)
    time.sleep(distance / speed / 3)
    tello.send_rc_control(-speed, 0, int(-speed*1.5), 0)
    time.sleep(distance / speed / 3)
    tello.send_rc_control(0, 0, -speed, 0)
    time.sleep(distance / speed / 3)
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(distance / speed / 3)

def curve_down_right(tello, distance, speed):
    tello.send_rc_control(speed, 0, -speed, 0)
    time.sleep(distance / speed / 3)
    tello.send_rc_control(speed, 0, int(-speed*1.5), 0)
    time.sleep(distance / speed / 3)
    tello.send_rc_control(0, 0, -speed, 0)
    time.sleep(distance / speed / 3)
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(distance / speed / 3)

def draw_L(tello):
    print("[INFO] Drawing 'L' in the air...")

    # Drop down
    tello.send_rc_control(0, 0, -30, 0) 
    time.sleep(4)
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)

    # Move left
    tello.send_rc_control(-15, 0, 0, 0)  # x = -30 => left
    time.sleep(2)
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)


# ========================
# FrameGrabber
# ========================

class FrameGrabber:
    def __init__(self, tello):
        self.frame_read = tello.get_frame_read()
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            with self.lock:
                self.frame = self.frame_read.frame
            time.sleep(0.03)  # ~30 FPS

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()

# ========================
# Main Letter Model Logic
# ========================

def centre_crop(img):
    h, w, _ = img.shape
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return img[top:top+min_dim, left:left+min_dim]

def run_letter_model():
    print("[INFO] Running real-time classification. Press 'q' to quit model view.")

    l_start_time = None
    l_triggered = False

    try:
        while True:
            frame = grabber.read()
            if frame is None:
                continue

            cropped = centre_crop(frame)
            resized = cv2.resize(cropped, (224, 224))
            input_tensor = tf.convert_to_tensor([resized], dtype=tf.float32) / 255.0

            predictions = model(input_tensor)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            predicted_label = labels[predicted_class]

            # Display result
            if confidence > 0.6:
                label_text = f"{predicted_label} ({confidence:.2f})"
                cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

            # Require continuous confident 'L' for 3 seconds
            if predicted_label == "L" and confidence >= 0.8:
                if l_start_time is None:
                    l_start_time = time.time()
                elif not l_triggered and (time.time() - l_start_time) >= 3:
                    print("[ACTION] Detected 'L' confidently for 3 seconds. Executing movement...")
                    draw_L(tello)
                    tello.land()
                    l_triggered = True
                    break
            else:
                # If it's not confident L anymore, reset the timer completely
                l_start_time = None


            top3 = np.argsort(predictions[0])[-3:][::-1]
            print("Top predictions:")
            for i in range(3):
                idx = top3[i]
                print(f"  {labels[idx]} ({predictions[0][idx]:.2f})")

            cv2.imshow("Tello Letter Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Model view exited manually.")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted. Exiting model view...")

    finally:
        cv2.destroyWindow("Tello Letter Recognition")


# ========================
# Initialisation
# ========================

# Connect to drone
tello = Tello()
tello.connect()
print(f"[INFO] Battery level: {tello.get_battery()}%")

# Load model and labels
print("[INFO] Loading model...")
model = tf.keras.models.load_model("LetterRecognition/converted_keras/keras_model.h5")
labels = []
with open("LetterRecognition/converted_keras/labels.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            index, label = parts
            index = int(index)
            while len(labels) <= index:
                labels.append(None)
            labels[index] = label
print("[INFO] Model and labels loaded.")

# Start video stream and grabber
tello.streamon()
grabber = FrameGrabber(tello)

# SDK mode
tello.send_command_without_return("command")
time.sleep(2)

# Open control window
cv2.namedWindow("Tello Control")
cv2.imshow("Tello Control", 255 * np.ones((100, 300), dtype=np.uint8))
cv2.waitKey(1)

print("[INFO] Controls: T=Takeoff, L=Land, M=Model, Q=Quit, SPACE=Emergency Stop")

# ========================
# Keyboard Control Loop
# ========================

try:
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('t'):
            print("[INFO] Takeoff")
            tello.takeoff()
            time.sleep(2)
            tello.send_rc_control(0, 0, 30, 0)
            time.sleep(4)
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(2)


        elif key == ord('l'):
            print("[INFO] Landing")
            tello.land()
            time.sleep(2)

        elif key == 32:  # Spacebar
            print("[EMERGENCY] Landing immediately!")
            tello.emergency()
            break

        elif key == ord('m'):
            print("[INFO] Activating model...")
            run_letter_model()
            print("[INFO] Model session ended.")

        elif key == ord('q'):
            print("[INFO] Quit command received.")
            break

except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")

finally:
    grabber.stop()
    tello.end()
    cv2.destroyAllWindows()
