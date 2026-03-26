class NGRANReport:

    def __init__(
        self,
        ue_id: int,
        timestep: float,
        rsrp_values: dict[int, float],
        rsrq_values: dict[int, float],
    ):
        self.ue_id: int = ue_id
        self.rsrp_values: dict[int, float] = rsrp_values
        self.rsrq_values: dict[int, float] = rsrq_values
        self.timestep = timestep

    def __repr__(self):
        return f"NGRANReport(ue_id: {self.ue_id}, rsrp_values: {self.rsrp_values}, rsrq_values: {self.rsrq_values})"

    def __str__(self):
        return f"NGRANReport(ue_id: {self.ue_id}, rsrp_values: {self.rsrp_values}, rsrq_values: {self.rsrq_values})"
