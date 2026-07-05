from training.train import train
from main.config import (
    TIME_STEPS, MAX_EPISODE_STEPS, WIND_TRAIN, WIND_VAL, ALGORITHMS, SEEDS, SUFFIX, MODELS_DIR,
    RENDER_MODE,
)


def main(algos=ALGORITHMS, seeds=SEEDS, suffix=SUFFIX):
    print("\nSailboat Trim Optimization - DRL Training\n")

    for seed in seeds:
        for algorithm in algos:
            model_path = str(MODELS_DIR / f"sail_trim_{algorithm.lower()}_model_{TIME_STEPS}_{seed}{suffix}")
            metrics_csv = (
                f"results/metrics/results_{algorithm.lower()}_{TIME_STEPS}_{seed}{suffix}_{MAX_EPISODE_STEPS}.csv"
            )
            print("=" * 70)
            print(f"Training {algorithm} for {TIME_STEPS} steps (seed={seed}, suffix={suffix})")
            print("=" * 70)
            train(
                algorithm=algorithm,
                total_timesteps=TIME_STEPS,
                model_path=model_path,
                training_metrics_csv_path=metrics_csv,
                env_kwargs={"max_episode_steps": MAX_EPISODE_STEPS, "wind_csv_path": WIND_TRAIN,
                            "render_mode": RENDER_MODE},
                eval_wind_csv_path=WIND_VAL,
                seed=seed,
            )


if __name__ == "__main__":
    main()
