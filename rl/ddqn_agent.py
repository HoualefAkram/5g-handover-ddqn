import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore, Style, init

init(autoreset=True)
# --- Custom Imports ---
from data_models.latlng import LatLng
from rl.handover_env import HandoverEnv
from rl.replay_buffer import ReplayBuffer
from rl.checkpoint_manager import CheckpointManager
from utils.logger import Logger
from prepare import MAP_TOP_LEFT, MAP_BOTTOM_RIGHT, MCC, SEED, STEP_LENGTH


# ==========================================
# 1. NEURAL NETWORK ARCHITECTURE
# ==========================================


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.net(x)


def hard_update(target_net, policy_net):
    target_net.load_state_dict(policy_net.state_dict())


# ==========================================
# 2. INITIALIZATION & HYPERPARAMETERS
# ==========================================


USE_GPU = True
device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

env = HandoverEnv(
    top_left=MAP_TOP_LEFT,
    bottom_right=MAP_BOTTOM_RIGHT,
    mcc=MCC,
    step_len=STEP_LENGTH,
)

epoches = 500
lr = 1e-3
decay_val = 0.99
min_epsilon = 0.05
gamma = 0.99
update_rate = 100
batch_size = 32

policy_network = QNetwork().to(device)
target_network = QNetwork().to(device)
hard_update(target_network, policy_network)

criterion = nn.MSELoss()
adam = optim.Adam(policy_network.parameters(), lr=lr)

memory = ReplayBuffer()
checkpoint_manager = CheckpointManager()
tb_logger = Logger(logdir="outputs/runs")  # Initialize TensorBoard Writer

start_epoch, epsilon = checkpoint_manager.load_checkpoint(
    policy_network, target_network, adam, device=device, default_epsilon=1.0
)

# ==========================================
# 3. TRAINING LOOP
# ==========================================

counter = 0

print(f"--- Starting DDQN Training on {device} ---")
# To view your graphs, open a new terminal and run: tensorboard --logdir=outputs/runs
print(
    "TensorBoard is active! Run 'tensorboard --logdir=outputs/runs' to view progress."
)

for epoche in range(start_epoch, epoches):
    done = False
    state, _ = env.reset()

    # Episode tracking metrics for TensorBoard
    total_reward = 0
    step_count = 0
    ep_loss_sum = 0.0
    ep_loss_count = 0
    ep_max_q_sum = 0.0
    ep_max_q_count = 0
    ep_rsrp_sum = 0.0
    ep_rsrq_sum = 0.0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        # Epsilon-Greedy Action Selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_network(state_tensor)
                action = torch.argmax(q_values).item()
                # Track the max Q-value the network is predicting
                ep_max_q_sum += torch.max(q_values).item()
                ep_max_q_count += 1

        # Step Environment
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

        # Track raw RSRP/RSRQ if connected to a tower
        if env.agent.serving_bs and len(env.agent.generated_reports) > 0:
            last_report = env.agent.generated_reports[-1]
            serving_id = env.agent.serving_bs.id
            ep_rsrp_sum += last_report.rsrp_values.get(serving_id, 0)
            ep_rsrq_sum += last_report.rsrq_values.get(serving_id, 0)

        # Save to RAM buffer
        memory.append((state, action, reward, new_state, done))
        state = new_state

        # Train the network
        if len(memory) >= batch_size:
            batch = random.sample(memory.queue, batch_size)
            b_states, b_actions, b_rewards, b_new_states, b_dones = zip(*batch)

            b_states_t = torch.tensor(np.array(b_states), dtype=torch.float32).to(device)
            b_new_states_t = torch.tensor(np.array(b_new_states), dtype=torch.float32).to(device)
            b_actions_t = torch.tensor(b_actions, dtype=torch.int64).unsqueeze(1).to(device)
            b_rewards_t = torch.tensor(b_rewards, dtype=torch.float32).unsqueeze(1).to(device)
            b_dones_t = torch.tensor(b_dones, dtype=torch.float32).unsqueeze(1).to(device)

            with torch.no_grad():
                best_next_action_idxs = torch.argmax(
                    policy_network(b_new_states_t), dim=1, keepdim=True
                )
                target_optimal_next_qs = target_network(b_new_states_t)
                v_targets = target_optimal_next_qs.gather(1, best_next_action_idxs)
                bellman_targets = b_rewards_t + gamma * v_targets * (1 - b_dones_t)

            policy_preds = policy_network(b_states_t).gather(1, b_actions_t)
            loss = criterion(policy_preds, bellman_targets)

            adam.zero_grad()
            loss.backward()
            adam.step()

            # Track Loss for TensorBoard
            ep_loss_sum += loss.item()
            ep_loss_count += 1

            counter += 1
            if counter >= update_rate:
                counter = 0
                hard_update(target_network, policy_network)

    # Decay Epsilon
    epsilon = max(min_epsilon, epsilon * decay_val)

    # Save State
    memory.save()
    checkpoint_manager.save_checkpoint(
        epoche, epsilon, policy_network, target_network, adam
    )

    # ----------------------------------------------------
    # TENSORBOARD LOGGING (End of Epoch)
    # ----------------------------------------------------
    avg_loss = ep_loss_sum / ep_loss_count if ep_loss_count > 0 else 0.0
    avg_max_q = ep_max_q_sum / ep_max_q_count if ep_max_q_count > 0 else 0.0
    avg_rsrp = ep_rsrp_sum / step_count if step_count > 0 else 0.0
    avg_rsrq = ep_rsrq_sum / step_count if step_count > 0 else 0.0

    # Global Metrics
    tb_logger.log_global_metric(Logger.Metric.TOTAL_REWARD, total_reward, epoche)
    tb_logger.log_global_metric(Logger.Metric.EPISODE_LENGTH, step_count, epoche)
    tb_logger.log_global_metric(Logger.Metric.EPSILON, epsilon, epoche)
    tb_logger.log_global_metric(Logger.Metric.AVERAGE_LOSS, avg_loss, epoche)
    tb_logger.log_global_metric(Logger.Metric.AVERAGE_MAX_Q, avg_max_q, epoche)

    # UE Specific Metrics
    ue_id = env.agent.id
    tb_logger.log_ue_metric(
        ue_id, Logger.Metric.TOTAL_HANDOVERS, env.agent.get_total_handovers(), epoche
    )
    tb_logger.log_ue_metric(
        ue_id, Logger.Metric.TOTAL_PINGPONG, env.agent.get_total_pingpong(), epoche
    )
    tb_logger.log_ue_metric(
        ue_id, Logger.Metric.PINGPONG_RATE, env.agent.get_pingpong_rate(), epoche
    )
    tb_logger.log_ue_metric(ue_id, Logger.Metric.RSRP, avg_rsrp, epoche)
    tb_logger.log_ue_metric(ue_id, Logger.Metric.RSRQ, avg_rsrq, epoche)

    # --- Terminal Output ---
    current_epoch = epoche + 1
    percent_complete = (current_epoch / epoches) * 100

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Epoch {current_epoch}/{epoches} "
        f"{Fore.YELLOW}[{percent_complete:.1f}%] "
        f"{Fore.GREEN}| Reward: {total_reward:.2f} "
        f"{Fore.WHITE}| Loss: {avg_loss:.4f} "
        f"{Fore.BLUE}| Handovers: {env.agent.get_total_handovers()} "
        f"{Fore.RED}| Ping-Pongs: {env.agent.get_total_pingpong()}"
    )

# Close TensorBoard
tb_logger.close()

# --- EXPORT MODEL ---
print(f"{Fore.GREEN}{Style.BRIGHT}Training Complete! Exporting final model...")
torch.save(policy_network.state_dict(), "outputs/final_ddqn_model.pth")
