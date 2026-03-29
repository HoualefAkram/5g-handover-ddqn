from enum import Enum
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional


class Logger:
    class Metric(Enum):
        # Training
        EPISODE_LENGTH = "Episode_Length"
        TOTAL_REWARD = "Total_Reward"
        AVERAGE_MAX_Q = "Average_Max_Q"
        AVERAGE_LOSS = "Average_Loss"
        EPSILON = "Epsilon"

        # Test
        RSRP = "RSRP"
        RSRQ = "RSRQ"
        AVERAGE_RSRP = "AVERAGE_RSRP"
        AVERAGE_RSRQ = "AVERAGE_RSRQ"
        TOTAL_HANDOVERS = "TOTAL_HANDOVERS"
        AVERAGE_HANDOVERS = "AVERAGE_HANDOVERS"
        TOTAL_PINGPONG = "TOTAL_PINGPONG"
        AVERAGE_PINGPONG = "AVERAGE_PINGPONG"
        PINGPONG_RATE = "PINGPONG_RATE"
        AVERAGE_PINGPONG_RATE = "AVERAGE_PINGPONG_RATE"

    def __init__(self, name: Optional[str], logdir: str = "outputs/runs"):
        prefix = f"{name}_" if name is not None else ""
        run_name = f'{prefix}{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        run_dir = Path(logdir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(run_dir))

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
