from pathlib import Path


def main() -> None:
    print(f"Run `uv run python src/train.py` from {Path.cwd()} to train the churn models.")


if __name__ == "__main__":
    main()
