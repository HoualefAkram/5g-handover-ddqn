import torch
from pathlib import Path


class CheckpointManager:
    def __init__(self, file_path: str = "cache/training/ddqn_checkpoint.pth"):
        self.file_path = Path(file_path)

    def save_checkpoint(
        self, epoch: int, epsilon: float, policy_net, target_net, optimizer
    ):
        """Saves the exact state of the networks, optimizer, and training progress."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Pack everything into a single PyTorch dictionary
        checkpoint = {
            "epoch": epoch,
            "epsilon": epsilon,
            "policy_net_state_dict": policy_net.state_dict(),
            "target_net_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(checkpoint, self.file_path)

    def load_checkpoint(
        self,
        policy_net,
        target_net,
        optimizer,
        default_epsilon: float = 1.0,
    ):
        """Loads the state into the models/optimizer and returns the current epoch and epsilon."""
        if self.file_path.exists() and self.file_path.is_file():
            # Load the dictionary back into RAM
            checkpoint = torch.load(self.file_path)

            # override the nets and optimizer with the saved values
            policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            target_net.load_state_dict(checkpoint["target_net_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            start_epoch = (
                checkpoint["epoch"] + 1
            )  # +1 so we don't repeat the last saved epoch
            epsilon = checkpoint["epsilon"]

            print(
                f"Checkpoint found... Resuming at Epoch {start_epoch} with Epsilon {epsilon:.3f}"
            )
            return start_epoch, epsilon
        else:
            print("No checkpoint found. Starting a fresh training session.")
            return 0, default_epsilon
