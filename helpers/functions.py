from math import exp, cos, radians


class Functions:
    @staticmethod
    def softmax(all_values: list[float], index: int) -> float:
        """Returns only the softmax output of the index"""
        max_val = max(all_values)
        numerator = exp(all_values[index] - max_val)
        denominator = sum(exp(v - max_val) for v in all_values)
        if denominator == 0:
            return 0
        return numerator / denominator

    @staticmethod
    def softmax_all(all_values: list[float]) -> list[float]:
        """Returns the softmax output of all_values"""
        if not all_values:
            return []

        max_val = max(all_values)
        exps = [exp(v - max_val) for v in all_values]
        denominator = sum(exps)

        return [e / denominator for e in exps]

    @staticmethod
    def cos_similarity(angle1: float, angle2: float) -> float:
        """
        Returns similarity shifted to [0.0, 1.0].
        1.0 = Driving directly towards
        0.5 = Perpendicular
        0.0 = Driving directly away
        """
        raw_cos = cos(radians(angle1 - angle2))
        return (raw_cos + 1.0) / 2.0

    @staticmethod
    def weighted_sum(values: list[float], weights: list[float]):
        assert len(values) == len(weights), "values must be the same length as weights."
        s = 0
        for i in range(len(values)):
            s += values[i] * weights[i]
        return s
