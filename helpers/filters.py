from data_models.base_tower import BaseTower
from data_models.ng_ran_report import NGRANReport
from utils.wave_utils import WaveUtils


class Filters:

    @staticmethod
    def top_k_towers(
        all_bs: list[BaseTower],
        report: NGRANReport,
        k: int = 4,
    ):

        scores: dict[int, float] = {}

        for bs in all_bs:
            bs_id = bs.id

            scores[bs_id] = WaveUtils.normalize_rsrp_index(
                report.rsrp_values.get(bs_id, 0), bs.radio
            )

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        top_k_ids = [item[0] for item in sorted_scores[:k]]

        tower_lookup = {bs.id: bs for bs in all_bs}
        top_k_towers = [tower_lookup[t_id] for t_id in top_k_ids]

        return top_k_towers
