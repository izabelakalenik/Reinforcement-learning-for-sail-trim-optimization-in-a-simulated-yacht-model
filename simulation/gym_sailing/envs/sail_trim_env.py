from pathlib import Path
from collections import deque
import gymnasium as gym
import numpy as np
import pandas as pd
from main.config import DEFAULT_POLAR_CSV, DEFAULT_WIND_CSV, FIXED_HEADING
from data.polar_data_preprocessing import build_polar_interpolator_from_csv
from simulation.gym_sailing.physics.sailboat import SailBoat
from simulation.gym_sailing.utils.angles import norm
from simulation.gym_sailing.utils.renderer import Renderer
from training.mdp import SailTrimMDP
from training.reward import compute_trim_reward


class SailTrimEnv(gym.Env):
    BOAT_BEAM = 1.4
    BOAT_LENGTH = 4.2
    COURSE_SIZE = 50
    TAIL_METRICS_WINDOW = 100

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, polar_csv_path=None, wind_csv_path=None, max_episode_steps=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = None
        self.mdp = SailTrimMDP(max_episode_steps=max_episode_steps) if max_episode_steps is not None else SailTrimMDP()
        
        self.action_space = self.mdp.action_space()
        self.observation_space = self.mdp.observation_space()

        self.polar_csv_path = Path(polar_csv_path) if polar_csv_path is not None else DEFAULT_POLAR_CSV
        self.wind_csv_path = Path(wind_csv_path) if wind_csv_path is not None else DEFAULT_WIND_CSV
        self.polar_interpolator = build_polar_interpolator_from_csv(str(self.polar_csv_path), output_unit='m/s')
        self.wind_profile = self._load_wind_profile(self.wind_csv_path)

        self.boat = None
        self.stepnum = 0

        self.step_reward = 0.0
        self.step_action = 0.0
        self.step_trim_efficiency = 0.0

        self.episode_reference_speed = 0.0
        self.episode_wind_speed = 0.0
        self.episode_wind_vector = np.array([0.0, 0.0], dtype=float)
        self.episode_wind_relative_heading = 0.0          # magnitude |TWA| (polar/metrics)
        self.episode_wind_relative_heading_signed = 0.0   # signed TWA (observation)
        self.episode_wind_direction = None
        self.episode_counter = 0
        
        # per-episode accumulators for summary stats
        self._ep_sum_boat_speed = 0.0
        self._ep_sum_reward = 0.0
        self._ep_sum_trim_efficiency = 0.0
        self._ep_steps = 0
        self._ep_tail_boat_speeds = deque(maxlen=self.TAIL_METRICS_WINDOW)
        
        # last episode info (for logging)
        self._last_episode_info = {}

    def _load_wind_profile(self, csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        required_columns = {"wind_speed", "wind_direction"}
        if not required_columns.issubset(df.columns):
            missing = ", ".join(sorted(required_columns.difference(df.columns)))
            raise ValueError(f"Wind CSV is missing required columns: {missing}")

        if "valid_time" in df.columns:
            df["valid_time"] = pd.to_datetime(df["valid_time"])
            df = df.sort_values("valid_time")

        df = df.reset_index(drop=True)

        # remove rows whose polar reference speed is below the minimum useful threshold,
        # with fixed heading pi/2, ref_speed~=0 (no-go zone) produces a degenerate reward
        # independent of sail angle - wasted training steps with no learning signal
        boat_heading = FIXED_HEADING

        def _ref_speed(row):
            wind_vec = self._wind_vector_from_sample(row["wind_speed"], row["wind_direction"])
            wind_speed = float(np.linalg.norm(wind_vec))
            if wind_speed <= 1e-6:
                return 0.0
            wind_from_angle = float(np.arctan2(-wind_vec[1], -wind_vec[0]))
            twa = float(abs(norm(wind_from_angle - boat_heading)))
            twa_deg = float(np.clip(np.degrees(twa), 0.0, 180.0))
            return float(self.polar_interpolator(twa_deg, wind_speed))

        ref_speeds = df.apply(_ref_speed, axis=1)
        before = len(df)
        df = df[ref_speeds >= self.mdp.min_ref_speed].reset_index(drop=True)
        print(f"Wind profile: {before} rows -> {len(df)} after filtering ref_speed < {self.mdp.min_ref_speed} m/s")

        return df

    def _wind_vector_from_sample(self, wind_speed: float, wind_direction: float) -> np.ndarray:
        direction_rad = np.deg2rad(float(wind_direction))
        wind_u = -float(wind_speed) * np.sin(direction_rad)
        wind_v = -float(wind_speed) * np.cos(direction_rad)
        return np.array([wind_u, wind_v], dtype=float)

    def _set_wind_from_row(self, row: pd.Series):
        wind_speed = float(row["wind_speed"])
        self.episode_wind_vector = self._wind_vector_from_sample(wind_speed, float(row["wind_direction"]))
        self.episode_wind_direction = float(row["wind_direction"])

    def _polar_boat_speed(self, wind_relative_heading: float, wind_speed: float) -> float:
        if self.polar_interpolator is None:
            return 0.0

        twa_deg = float(np.degrees(abs(wind_relative_heading)))
        twa_deg = float(np.clip(twa_deg, 0.0, 180.0))
        polar_speed = self.polar_interpolator(twa_deg, float(wind_speed))
        return float(polar_speed)

    def _episode_polar_reference(self):
        wind_vector = self.episode_wind_vector
        wind_speed = float(np.linalg.norm(wind_vector))
        if wind_speed <= 1e-6:
            return 0.0, 0.0, 0.0, wind_speed

        wind_from_angle = float(np.arctan2(-wind_vector[1], -wind_vector[0]))
        signed_relative_heading = float(norm(wind_from_angle - self.boat.heading))
        wind_relative_heading = abs(signed_relative_heading)
        reference_speed = self._polar_boat_speed(wind_relative_heading, wind_speed)
        return (
            reference_speed,
            float(np.degrees(wind_relative_heading)),
            float(np.degrees(signed_relative_heading)),
            wind_speed,
        )

    def reset(self, options=None, seed=None):
        """
        Start a new episode with randomized wind and cleared episode state.

        This method reinitializes the boat at the center of the course, samples
        initial wind direction and strength, resets reward/action history, and
        returns the initial observation expected by Gymnasium.
        """

        super().reset(seed=seed)

        initial_speed = float(self.np_random.uniform(
            self.mdp.initial_speed_low, self.mdp.initial_speed_high
        ))
        self.boat = SailBoat(
            x=self.COURSE_SIZE * 0.5,
            y=self.COURSE_SIZE * 0.5,
            heading=FIXED_HEADING,
            heading_dot=0.0,
            speed=initial_speed,
        )

        if self.wind_profile.empty:
            raise ValueError("Wind profile CSV is empty.")
        
        random_wind_row_index = int(self.np_random.integers(len(self.wind_profile)))
        wind_row = self.wind_profile.iloc[random_wind_row_index]
        self._set_wind_from_row(wind_row)

        polar_speed, wind_relative_heading_deg, wind_relative_heading_signed_deg, wind_speed = self._episode_polar_reference()
        self.episode_reference_speed = float(polar_speed)
        self.episode_wind_speed = float(wind_speed)
        self.episode_wind_relative_heading = float(wind_relative_heading_deg)
        self.episode_wind_relative_heading_signed = float(wind_relative_heading_signed_deg)

        self.stepnum = 0
        self.step_reward = 0.0
        self.step_action = 0.0
        self.step_trim_efficiency = 0.0

        # reset per-episode accumulators at episode start
        self._ep_sum_boat_speed = 0.0
        self._ep_sum_reward = 0.0
        self._ep_sum_trim_efficiency = 0.0
        self._ep_steps = 0
        self._ep_tail_boat_speeds.clear()

        if self.render_mode in ["human", "rgb_array"]:
            self.renderer = Renderer(self.BOAT_LENGTH, self.BOAT_BEAM, self.COURSE_SIZE)

        self.episode_counter += 1

        return self._get_obs(), {}

    def step(self, action):
        self.stepnum += 1
        sail_angle = np.clip(float(action[0]), -np.pi / 2, np.pi / 2)
        self.step_action = sail_angle
        self.boat.set_sail_angle(sail_angle)

        # physics step - no acceleration returned now
        self.boat.step_trim_only(self.episode_wind_vector)

        obs = self._get_obs()
        
        # compute reward using SPEED-BASED objective function
        reward, reward_breakdown = compute_trim_reward(
            current_boat_speed=self.boat.speed,
            reference_speed=self.episode_reference_speed,
            speed_max_coefficient=self.mdp.speed_max_coefficient,
            speed_tracking_coefficient=self.mdp.speed_tracking_coefficient,
            step_penalty=self.mdp.step_penalty,
        )
        
        self.step_trim_efficiency = float(reward_breakdown["trim_efficiency"])
        self.step_reward = reward

        # accumulate per-episode stats
        self._ep_sum_boat_speed += float(self.boat.speed)
        self._ep_sum_reward += float(reward)
        self._ep_sum_trim_efficiency += float(self.step_trim_efficiency)
        self._ep_steps += 1
        self._ep_tail_boat_speeds.append(float(self.boat.speed))

        if self._ep_steps > 0:
            episode_avg_boat_speed = self._ep_sum_boat_speed / self._ep_steps
            episode_avg_reward = self._ep_sum_reward / self._ep_steps
            episode_avg_sail_trim_efficiency = self._ep_sum_trim_efficiency / self._ep_steps
        else:
            episode_avg_boat_speed = 0.0
            episode_avg_reward = 0.0
            episode_avg_sail_trim_efficiency = 0.0

        if self._ep_tail_boat_speeds:
            episode_tail_avg_boat_speed = float(np.mean(self._ep_tail_boat_speeds))
            episode_tail_max_boat_speed = float(np.max(self._ep_tail_boat_speeds))
        else:
            episode_tail_avg_boat_speed = 0.0
            episode_tail_max_boat_speed = 0.0

        episode_final_boat_speed = float(self.boat.speed)
        episode_tail_avg_speed_error = abs(episode_tail_avg_boat_speed - self.episode_reference_speed)
        episode_final_speed_error = abs(episode_final_boat_speed - self.episode_reference_speed)
        episode_tail_trim_efficiency = episode_tail_avg_boat_speed / self.episode_reference_speed if self.episode_reference_speed > 1e-6 else 0.0

        terminated = False  # episode ends only on natural termination conditions (not applicable here)
        truncated = self.stepnum >= self.mdp.max_episode_steps  # episode ends when max timesteps reached (this case)

        if self.render_mode == "human":
            self._render_frame()

        # if episode ended, print a short per-episode summary
        if terminated or truncated:
            print(
                f"[Episode {self.episode_counter}]\n"
                f"  wind_dir={self.episode_wind_direction:.1f}deg\n"
                f"  wind_speed={self.episode_wind_speed:.2f}\n"
                f"  twa={self.episode_wind_relative_heading:.1f}deg\n"
                f"  ref_speed={self.episode_reference_speed:.3f}\n"
                f"  episode_avg_boat_speed={episode_avg_boat_speed:.3f}\n"
                f"  episode_tail_avg_boat_speed={episode_tail_avg_boat_speed:.3f}\n"
                f"  episode_tail_max_boat_speed={episode_tail_max_boat_speed:.3f}\n"
                f"  episode_final_boat_speed={episode_final_boat_speed:.3f}\n"
                f"  episode_tail_avg_speed_error={episode_tail_avg_speed_error:.3f}\n"
                f"  episode_final_speed_error={episode_final_speed_error:.3f}\n"
                f"  episode_avg_sail_trim_efficiency={episode_avg_sail_trim_efficiency:+.3f}\n"
                f"  episode_avg_reward={episode_avg_reward:+.3f}\n"
            )

        info = {
            "wind_relative_heading": self.episode_wind_relative_heading,
            "wind_speed": self.episode_wind_speed,
            "wind_direction": self.episode_wind_direction,
            "episode_reference_speed": self.episode_reference_speed,
            "episode_avg_boat_speed": float(episode_avg_boat_speed),
            "episode_tail_avg_boat_speed": float(episode_tail_avg_boat_speed),
            "episode_tail_max_boat_speed": float(episode_tail_max_boat_speed),
            "episode_final_boat_speed": float(episode_final_boat_speed),
            "episode_tail_avg_speed_error": float(episode_tail_avg_speed_error),
            "episode_final_speed_error": float(episode_final_speed_error),
            "episode_tail_trim_efficiency": float(episode_tail_trim_efficiency),
            "episode_avg_sail_trim_efficiency": float(episode_avg_sail_trim_efficiency),
            "episode_avg_reward": float(episode_avg_reward),
            "episode_counter": self.episode_counter,
        }
        
        # store episode info for logging callback (callback tracks episode_counter change)
        self._last_episode_info = info
        
        return obs, float(reward), terminated, truncated, info

    # state observation
    def _get_obs(self):
        return np.array(
            [self.episode_wind_speed, np.deg2rad(self.episode_wind_relative_heading_signed),
             self.boat.sail_angle, self.boat.speed],
            dtype=np.float32,
        )

    def _render_frame(self):
        wind_vector = self.episode_wind_vector
        rendered = self.renderer.render_frame(
            boats=[(self.boat.x, self.boat.y, self.boat.heading + np.pi / 2, self.step_action)],
            stepnum=self.stepnum,
            reward=self.step_reward,
            render_mode=self.render_mode,
            fps=self.metadata["render_fps"],
            wind_vector=wind_vector,
            boat_speed=self.boat.speed,
            trim_efficiency=self.step_trim_efficiency,
            reference_speed=self.episode_reference_speed,
            wind_speed=self.episode_wind_speed,
            wind_relative_heading=self.episode_wind_relative_heading,
        )

        if self.render_mode == "human":
            return None
        if self.render_mode == "rgb_array":
            import pygame

            return pygame.surfarray.array3d(self.renderer.window).transpose(1, 0, 2)
        return rendered

    def render(self):
        if self.render_mode == "rgb_array" and self.renderer is not None:
            return self._render_frame()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
