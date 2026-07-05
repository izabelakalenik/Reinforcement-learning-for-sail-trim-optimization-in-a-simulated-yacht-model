import numpy as np

class Boat:
    TIME_STEP = 0.1  # seconds

    def __init__(self, x, y, heading, heading_dot=0.0, speed=0.0):
        self.x = x
        self.y = y
        self.heading = heading
        self.heading_dot = heading_dot
        self.speed = speed
        self.velocity = speed * np.array([np.cos(heading), np.sin(heading)])
        self.mass = 6970.0  # kg

    def _update_state(self, value, delta_value):
        value += Boat.TIME_STEP * delta_value
        return value
