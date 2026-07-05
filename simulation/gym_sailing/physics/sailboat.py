import numpy as np
from simulation.gym_sailing.physics.boat import Boat
from simulation.gym_sailing.utils.angles import unit_vector, perpendicular, norm


class SailBoat(Boat):
    SAILCOEFF = 31.4         
    CL_MAX = 1.5             
    CD_MAX = 2.5             
    DRAGCOEFF = 13.55        
    WAVE_DRAGCOEFF = 3.384 

    def __init__(self, x, y, heading, heading_dot=0.0, speed=0.0):
        super().__init__(x, y, heading, heading_dot, speed)
        self.sail_angle = 0.0 

    def set_sail_angle(self, sail_angle):
        self.sail_angle = np.clip(sail_angle, -np.pi / 2, np.pi / 2)

    def step_trim_only(self, wind_vector):
        apparent_wind = wind_vector - self.velocity
        apparent_wind_speed = np.linalg.norm(apparent_wind)

        if apparent_wind_speed < 0.01:
            self.velocity *= 0.98
            self.x = self._update_state(self.x, self.velocity[0])
            self.y = self._update_state(self.y, self.velocity[1])
            self.speed = np.linalg.norm(self.velocity)
            return

        unit_heading = unit_vector(self.heading)
        wind_unit = apparent_wind / apparent_wind_speed
        wind_perp = perpendicular(wind_unit)  

        # signed angle of attack between apparent wind and the sail chord, folded to
        # [-pi/2, pi/2] because the sail is a symmetric line (both ends equivalent)
        wind_dir = np.arctan2(apparent_wind[1], apparent_wind[0])
        angle_of_attack = norm(self.heading + self.sail_angle - wind_dir)
        if angle_of_attack > np.pi / 2:
            angle_of_attack -= np.pi
        elif angle_of_attack < -np.pi / 2:
            angle_of_attack += np.pi

        lift_coeff = self.CL_MAX * np.sin(2.0 * angle_of_attack)  # signed: sets lift side
        drag_coeff = self.CD_MAX * np.sin(angle_of_attack) ** 2   # always >= 0

        aero_force = self.SAILCOEFF * apparent_wind_speed ** 2 * (
            lift_coeff * wind_perp + drag_coeff * wind_unit
        )

        forward_force = max(0.0, float(np.dot(aero_force, unit_heading))) * unit_heading

        speed_magnitude = np.linalg.norm(self.velocity)
        drag_force = -(self.DRAGCOEFF * speed_magnitude
                       + self.WAVE_DRAGCOEFF * speed_magnitude ** 3) * self.velocity

        total_force = forward_force + drag_force
        acceleration = total_force / self.mass
        self.velocity += acceleration * self.TIME_STEP
        self.x = self._update_state(self.x, self.velocity[0])
        self.y = self._update_state(self.y, self.velocity[1])
        self.speed = np.linalg.norm(self.velocity)
