from math import exp, cos


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
        return [
            Functions.softmax(all_values=all_values, index=i)
            for i in range(len(all_values))
        ]

    @staticmethod
    def cos_similarity(angle1: float, angle2: float):
        return cos(angle1 - angle2)

    @staticmethod
    def weighted_sum(values: list[float], weights: list[float]):
        assert len(values) == len(weights), "values must be the same length as weights."
        s = 0
        for i in range(len(values)):
            s += values[i] * weights[i]
