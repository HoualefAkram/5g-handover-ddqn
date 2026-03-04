class NGRANReport:

    def __init__(
        self,
        ue_id: int,
        rsrp_values: dict,
        rsrq_values: dict,
    ):
        self.ue_id: int = ue_id
        self.rsrp_values: dict = rsrp_values
        self.rsrq_values: dict = rsrq_values

    def __repr__(self):
        return f"NGRANReport(ue_id: {self.ue_id}, rsrp_values: {self.rsrp_values}, rsrq_values: {self.rsrq_values})"

    def __str__(self):
        return f"NGRANReport(ue_id: {self.ue_id}, rsrp_values: {self.rsrp_values}, rsrq_values: {self.rsrq_values})"
