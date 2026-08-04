import argparse
import logging
from omni_med_agent.settings import load_settings
from omni_med_agent.training.seed import set_seed

def main() -> None:
    parser = argparse.ArgumentParser(prog="omnimed-train")
    parser.add_argument("--config", required=True)
    settings = load_settings(parser.parse_args().config)
    logging.basicConfig(level=logging.INFO)
    set_seed(settings.train.seed)
    logging.getLogger(__name__).info("stage=%s steps=%d", settings.train.stage, settings.train.total_steps)

if __name__ == "__main__":
    main()
