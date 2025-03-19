# make drone draw letter 'L' in the air

import cv2 as cv
import numpy as np
import threading
import time
from djitellopy import Tello

def draw_L():

    # move down 
    tello.move_down(50)
    time.sleep(5)
    
    # move left
    tello.move_left(50)
    
def diagonal_up_right(tello, distance=200, speed=30):
    """
    Moves the drone diagonally up and to the right.

    :param tello: Tello drone object
    :param distance: Distance to move in cm (default 50)
    :param speed: Speed of movement (default 30)
    """
    tello.send_rc_control(speed, speed, speed, 0)  # Move diagonally
    time.sleep(distance / speed)  # Calculate time dynamically
    tello.send_rc_control(0, 0, 0, 0)  # Stop movement
# Initialize Tello
tello = Tello()
tello.connect()
print(tello.get_battery())

# Ensure SDK mode is properly activated
tello.send_command_without_return("command")
time.sleep(5)  # Wait for SDK mode

# Take off and wait before moving
tello.takeoff()
time.sleep(5)

diagonal_up_right(tello)

tello.land()

    


