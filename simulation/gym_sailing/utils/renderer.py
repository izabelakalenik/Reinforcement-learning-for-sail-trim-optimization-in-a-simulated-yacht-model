import os
import numpy as np
import pygame
from simulation.gym_sailing.utils.angles import norm


class Renderer:
    WATER_COLOR = (38, 102, 138)
    BOAT_COLOR = (220, 245, 230)
    INFO_COLOR = (102, 160, 198)

    def __init__(self, boat_length=4.2, boat_beam=1.4, course_size=60):
        self.boat_length = boat_length
        self.boat_beam = boat_beam
        self.course_size = course_size

        self.screen_width = 850
        self.scale = self.screen_width / self.course_size
        self.screen_height = int(self.scale * self.course_size)

        pygame.font.init()
        self.normal_font = pygame.font.SysFont("monospace", 20)

        boatwidth = self.boat_beam * self.scale
        boatlength = self.boat_length * self.scale
        path = os.path.dirname(os.path.abspath(__file__))

        self.boat_img = pygame.image.load(os.path.join(path, "assets/hull.png"))
        self.boat_img = pygame.transform.scale(self.boat_img, (boatwidth * 30, boatlength * 30))

        self.sail_img = pygame.image.load(os.path.join(path, "assets/sail.png"))
        self.sail_img = pygame.transform.scale(self.sail_img, (boatwidth * 10, boatlength * 15))

        self.window = None
        self.clock = None

    def render_frame(
        self,
        boats,
        stepnum,
        reward,
        render_mode,
        fps,
        wind_vector=None,
        boat_speed=None,
        trim_efficiency=None,
        reference_speed=None,
        wind_speed=None,
        wind_relative_heading=None,
    ):
        if self.window is None and render_mode in ["human", "rgb_array"]:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Sailboat Trim Optimization")

        if self.clock is None and render_mode == "human":
            self.clock = pygame.time.Clock()

        self.draw_water()

        for boat in boats:
            self.draw_boat((boat[0], boat[1]), boat[2])

        self.window.blit(pygame.transform.flip(self.window, False, True), (0, 0))

        self.render_metrics(
            wind_vector=wind_vector,
            stepnum=stepnum,
            boat_speed=boat_speed if boat_speed is not None else 0.0,
            reward=reward,
            trim_efficiency=trim_efficiency,
            reference_speed=reference_speed,
            wind_speed=wind_speed,
        )
        
        if wind_vector is not None:
            self.draw_wind_arrow(wind_vector)

        if render_mode == "human":
            pygame.display.flip()
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
            self.clock.tick(fps)
        elif render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.window).transpose(1, 0, 2)

    def _format_trim_metrics(self, reference_speed=None, boat_speed=None, trim_efficiency=None, wind_speed=None):
        details_parts = []
        if reference_speed is not None:
            details_parts.append(f"ref speed: {reference_speed:4.2f}")
        if boat_speed is not None:
            details_parts.append(f"boat speed: {boat_speed:4.2f}")
        if trim_efficiency is not None:
            details_parts.append(f"trim eff: {trim_efficiency:4.2f}")
        if wind_speed is not None:
            details_parts.append(f"wind speed: {wind_speed:4.2f}")
        return " | ".join(details_parts) if details_parts else None

    def render_metrics(
        self,
        wind_vector,
        stepnum,
        boat_speed,
        reward,
        trim_efficiency=None,
        reference_speed=None,
        wind_speed=None,
    ):
        details = self._format_trim_metrics(
            reference_speed=reference_speed,
            boat_speed=boat_speed,
            trim_efficiency=trim_efficiency,
            wind_speed=wind_speed,
        )

        if self.window is None:
            return

        info_label = self.normal_font.render(f"step:{stepnum:4} reward:{reward:+4.2f}", True, self.INFO_COLOR)
        self.window.blit(info_label, (7, self.screen_height - 25))

        if details is None:
            return

        details_label = self.normal_font.render(details, True, self.INFO_COLOR)
        self.window.blit(details_label, (7, self.screen_height - 48))

    def draw_water(self):
        self.window.fill(self.WATER_COLOR)

    def draw_boat(self, boat_pos, boat_heading):
        delta = (-np.array([np.sin(boat_heading), np.cos(-boat_heading)]) * 0.15 * self.boat_length * self.scale)
        self.draw_hull(boat_pos, boat_heading)
        self.draw_sail(boat_pos, boat_heading, delta)

    def _wrap_position(self, x_pos, y_pos, margin=100):
        x_pos = (x_pos + margin) % (self.screen_width + 2 * margin) - margin
        y_pos = (y_pos + margin) % (self.screen_height + 2 * margin) - margin
        return x_pos, y_pos

    def draw_hull(self, boat_pos, boat_heading):
        boat_img = pygame.transform.rotozoom(self.boat_img, np.degrees(-boat_heading), 0.06)

        x_pos = int(boat_pos[0] * self.scale)
        y_pos = int(boat_pos[1] * self.scale)
        x_pos, y_pos = self._wrap_position(x_pos, y_pos)

        boat_rect = boat_img.get_rect()
        boat_rect.center = (x_pos, y_pos)

        boat_img.fill(self.BOAT_COLOR, special_flags=pygame.BLEND_RGB_MULT)
        self.window.blit(boat_img, boat_rect)

    def draw_sail(self, boat_pos, boat_heading, delta):
        norm_heading = norm(boat_heading)

        pos_x = boat_pos[0] * self.scale + delta[0]
        pos_y = boat_pos[1] * self.scale - delta[1]
        pos_x, pos_y = self._wrap_position(pos_x, pos_y)

        if abs(norm_heading) < 0.5:
            end_x = int(boat_pos[0] * self.scale)
            end_y = int((boat_pos[1] - self.boat_length * 0.5) * self.scale)
            end_x, end_y = self._wrap_position(end_x, end_y)

            pygame.draw.aaline(self.window, (0, 0, 0), (pos_x, pos_y), (end_x, end_y))
            return

        if norm_heading > 0:
            sail_img = pygame.transform.flip(self.sail_img, True, False)
            sail_img = pygame.transform.rotozoom(
                sail_img, np.degrees(-0.45 * (norm_heading + 0.92)) + 90, 0.10
            )
        else:
            sail_img = pygame.transform.rotozoom(
                self.sail_img.copy(), np.degrees(-0.45 * (norm_heading - 0.92)) + 90, 0.10
            )

        self.window.blit(sail_img, sail_img.get_rect(center=(int(pos_x), int(pos_y))))

    def draw_wind_arrow(self, wind_vector, start_pos=None):
        if start_pos is None:
            start_pos = (self.screen_width - 100, 115)

        wind_magnitude = np.linalg.norm(wind_vector)
        if wind_magnitude < 0.01:
            return

        wind_dir = wind_vector / wind_magnitude
        arrow_length = 70
        screen_dir = np.array([wind_dir[0], -wind_dir[1]])
        arrow_end = (start_pos[0] + wind_dir[0] * arrow_length, start_pos[1] - wind_dir[1] * arrow_length)

        pygame.draw.line(self.window, (255, 100, 0), start_pos, arrow_end, width=4)

        wind_label = self.normal_font.render("WIND", True, (255, 100, 0))
        self.window.blit(wind_label, (start_pos[0] - 25, start_pos[1] - 105))

        arrowhead_length = 18
        arrowhead_width = 12
        perp = np.array([-screen_dir[1], screen_dir[0]])
        arrow_tip = np.array(arrow_end)
        base_center = arrow_tip - screen_dir * arrowhead_length
        arrowhead_pt1 = base_center + perp * arrowhead_width
        arrowhead_pt2 = base_center - perp * arrowhead_width

        pygame.draw.polygon(self.window, (255, 100, 0), [arrow_tip, arrowhead_pt1, arrowhead_pt2])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

