# make drone draw letter 'L' in the air

import cv2 as cv2
import numpy as np
import threading
import time
import keyboard
from djitellopy import Tello

# def draw_L():

#     # move down 
#     tello.move_down(50)
#     time.sleep(5)
    
#     # move left
#     tello.move_left(50)
    

def diagonal_up_right(tello, distance=100, speed=30):
    """
    Moves the drone diagonally up and to the right.

    :param tello: Tello drone object
    :param distance: Distance to move in cm (default 50)
    :param speed: Speed of movement (default 30)
    """
    
    # We got the sleep function from chatGPT as we were struggling to find a way to calculate the time dynamically as our seconds method was too slow and working inconsistently
    # This works by calculating the time it takes to move the drone a certain distance at a certain speed and then sleeping for that time we then divide that by 3 as each movement is broken down into 3 small parts for each curve.
    
    # send_rc_control(left/right, forward/backward, up/down, yaw)
    tello.send_rc_control(speed, 0, speed, 0)  # Move diagonally
    time.sleep(distance / speed)  # Calculate time dynamically
    tello.send_rc_control(0, 0, 0, 0)  # Stop movement
    
def curve_up_right(tello, distance, speed):
    # send_rc_control(left/right, forward/backward, up/down, yaw)
    diagonal_up_right(tello, distance, speed)
    diagonal_up_right(tello, distance, int(speed*1.5))
    tello.send_rc_control(0, 0, speed, 0)  # Move up
    time.sleep(distance / speed / 3)  # Calculate time dynamically
    tello.send_rc_control(0, 0, 0, 0)  # Stop movement
    time.sleep(distance / speed / 3)
    
def curve_up_left(tello, distance, speed):
    # send_rc_control(left/right, forward/backward, up/down, yaw)
    tello.send_rc_control(0, 0, speed, 0)  # Move up
    time.sleep(distance / speed / 3)  # Calculate time dynamically
    tello.send_rc_control(-speed, 0, int(speed*1.5), 0)  # Move diagonally
    time.sleep(distance / speed/ 3)  # Calculate time dynamically
    tello.send_rc_control(-speed, 0, speed, 0)  # Move diagonally
    time.sleep(distance / speed / 3)  # Calculate time dynamically
    tello.send_rc_control(0, 0, 0, 0)  # Stop movement
    time.sleep(distance / speed / 3)
    
def curve_down_left(tello, distance, speed):
    # send_rc_control(left/right, forward/backward, up/down, yaw)
    tello.send_rc_control(-speed, 0, -speed, 0)  # Move diagonally
    time.sleep(distance / speed / 3)  # Calculate time dynamically
    tello.send_rc_control(-speed, 0, int(-speed*1.5), 0)  # Move diagonally
    time.sleep(distance / speed/ 3)  # Calculate time dynamically
    tello.send_rc_control(0, 0, -speed, 0)  # Move down
    time.sleep(distance / speed / 3)  # Calculate time dynamically
    tello.send_rc_control(0, 0, 0, 0)  # Stop movement
    time.sleep(distance / speed / 3)
    
def curve_down_right(tello, distance, speed):
    # send_rc_control(left/right, forward/backward, up/down, yaw)
    tello.send_rc_control(speed, 0, -speed, 0)  # Move diagonally
    time.sleep(distance / speed / 3)  # Calculate time dynamically
    tello.send_rc_control(speed, 0, int(-speed*1.5), 0)  # Move diagonally
    time.sleep(distance / speed / 3)  # Calculate time dynamically
    tello.send_rc_control(0, 0, -speed, 0)  # Move down
    time.sleep(distance / speed / 3)  # Calculate time dynamically
    tello.send_rc_control(0, 0, 0, 0)  # Stop movement
    time.sleep(distance / speed / 3)
    
def circle():
    curve_up_right(tello, 50, 50)
    curve_up_left(tello, 50, 50)
    curve_down_left(tello, 50, 50)
    curve_down_right(tello, 50, 50)


# Initialize Tello
tello = Tello()
tello.connect()
print(tello.get_battery())

# Ensure SDK mode is properly activated
tello.send_command_without_return("command")
time.sleep(5)  # Wait for SDK mode

# Create dummy OpenCV window for key capture
cv2.namedWindow("Tello Control")
cv2.imshow("Tello Control", 255 * np.ones((100, 300), dtype=np.uint8))
cv2.waitKey(1)

print("[INFO] Press T to take off, L to land, SPACE for EMERGENCY STOP")

try:
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('t'):
            print("[INFO] Takeoff")
            tello.takeoff()
            time.sleep(2)

        elif key == ord('l'):
            print("[INFO] Landing")
            tello.land()
            time.sleep(2)

        elif key == 32:  # Spacebar
            print("[EMERGENCY] Landing immediately!")
            tello.emergency()
            break

        elif key == ord('q'):
            print("[INFO] Quit command received.")
            break

except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")

finally:
    tello.end()
    cv2.destroyAllWindows()