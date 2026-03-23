from enum import Enum
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


class Logger:
    class Metric(Enum):
        EPISODE_LENGTH = "Episode_Length"
        TOTAL_REWARD = "Total_Reward"
        AVERAGE_MAX_Q = "Average_Max_Q"
        AVERAGE_LOSS = "Average_Loss"
        RSRP = "RSRP"
        RSRQ = "RSRQ"
        EPSILON = "Epsilon"

    def __init__(self, logdir: str = "outputs/runs"):
        Path(logdir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(logdir)

    def log_ue_metric(self, ue_index: int, metric: Metric, value: float, step: int):
        tag = f"UE_{ue_index}/{metric.value}"
        self.writer.add_scalar(tag, value, step)

    def log_global_metric(
        self, metric: Metric, value: float, step: int, category: str = "Performance"
    ):
        tag = f"{category}/{metric.value}"
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()
