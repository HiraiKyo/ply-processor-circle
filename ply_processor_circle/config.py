from .utils.config import ConfigBase


class Config(ConfigBase):
    INLIER_THRESHOLD = 0.1
    MAX_ITERATION: int = 1000

    def load_config(self, config: dict) -> None:
        self.MODE = config.get("MODE", self.MODE)
        pass
