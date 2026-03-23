import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from data_models.base_tower import BaseTower
from data_models.car_fcd_data import CarFcdData
from data_models.handover_algorithm import HandoverAlgorithm
from data_models.ng_ran_report import NGRANReport
from data_models.user_equipment import UserEquipment
from utils.fcd_parser import FcdParser


class HandoverEnv(gym.Env):

    def __init__(
        self,
        base_towers: list[BaseTower],
        fcd_trace_file: str = "outputs/sumo/trace.xml",
    ):
        super().__init__()
        # UEs
        num_ue = FcdParser.count_vehicles(trace_file=fcd_trace_file)
        self.user_equipments = dict[int, UserEquipment] = {
            i: UserEquipment(
                id=i,
                all_bs=base_towers,
                print_report_on_movement=False,
                handover_algorithm=HandoverAlgorithm.DDQN_CHO,
            )
            for i in range(num_ue)
        }
        # Agent
        self.agent = self.user_equipments[0]
        # Base Towers
        self.base_towers = base_towers
        # action space: choosing 1 of 4 BS
        self.action_space = Discrete(4)
        # observation Space
        low = np.array([0] * 8 + [0] * 4)  # Min values for RSRP/RSRQ and One-Hot
        high = np.array([127] * 8 + [1] * 4)  # Max values
        self.observation_space = Box(low=low, high=high, dtype=np.int32)

        # simulation step by step
        self.timestep_idx = 0
        self.fcd_data = FcdParser.parse_fcd_trace(trace_file=fcd_trace_file)

    def step(self, action):
        """If the agent doesn't have a serving tower, execute the action instantly"""
        if not self.agent.serving_bs:
            self.agent.handover()  # TODO: transfer action to BS, NOTE: action is between 0 and 3, it doesnt have to do anything with BS ID
            # return state, reward, terminated, truncated, info

        """Compare Old Report with the newer one to generate the reward"""
        # get the newest report
        old_report: NGRANReport = self.agent.generated_reports[-1]
        old_rsrp_values = old_report.rsrp_values
        old_rsrq_values = old_report.rsrq_values

        # move all cars once
        fcds = self.fcd_data[self.timestep_idx].values()
        for fcd in fcds:
            car = self.user_equipments[fcd.id]
            report: NGRANReport = car.move_to(fcd.latlng, timestep=fcd.timestep)

        # get the updated report of the agent
        agent_report: NGRANReport = self.agent.generated_reports[-1]

        new_rsrp_values = agent_report.rsrp_values
        new_rsrp_values = agent_report.rsrq_values

        self.timestep_idx += 1

        # return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
