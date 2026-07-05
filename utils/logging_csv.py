import csv
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback


class CSVLoggingCallback(BaseCallback):

    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "episode",
                "wind_speed",
                "twa",
                "ref_speed",
                "avg_boat_speed",
                "tail_avg_boat_speed",
                "tail_avg_speed_error",
                "tail_trim_efficiency",
                "avg_trim_efficiency",
                "avg_reward",
            ],
        )
        self.csv_writer.writeheader()
        self._global_episode = 0

    def _write_row(self, info: dict) -> None:
        self._global_episode += 1
        row = {
            "episode": self._global_episode,
            "wind_speed": info.get("wind_speed", 0),
            "twa": info.get("wind_relative_heading", 0),
            "ref_speed": info.get("episode_reference_speed", 0),
            "avg_boat_speed": info.get("episode_avg_boat_speed", 0),
            "tail_avg_boat_speed": info.get("episode_tail_avg_boat_speed", 0),
            "tail_avg_speed_error": info.get("episode_tail_avg_speed_error", 0),
            "tail_trim_efficiency": info.get("episode_tail_trim_efficiency", 0),
            "avg_trim_efficiency": info.get("episode_avg_sail_trim_efficiency", 0),
            "avg_reward": info.get("episode_avg_reward", 0),
        }
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")

        if infos is None or dones is None:
            return True

        for info, done in zip(infos, dones):
            if done:
                self._write_row(info)

        return True

    def _on_training_end(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()
