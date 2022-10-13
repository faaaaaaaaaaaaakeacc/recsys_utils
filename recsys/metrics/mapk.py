from base import BaseMetric
from typing import Dict
import numpy as np


class MeamAveragePrecisionK(BaseMetric):
    """Class for computation map@k."""

    def __init__(self, k: int):
        """Init MeamAveragePrecisionK.

        Parameters
        ----------
        k: int
            metrics hyper parameter.
        """
        super().__init__()
        self.k = k

    def _apk(self, pred: np.array, target: np.array, k: int) -> float:
        """Compute average precision at k.

        Parameters
        ----------
        pred:
        """
        if len(target) == 0:
            return 0
        if len(pred) >= k:
            predicted = pred[:k]

        ans, cnt = 0, 0
        s = set()
        tot = min(len(target), k)
        for i in range(len(pred)):
            if pred[i] not in s and pred[i] in target:
                cnt += 1
                ans += cnt / (i + 1)
        return ans / tot

    def __call__(self, pred: Dict[int, np.array], target: Dict[int, np.array]) -> float:
        """Compute metric.

        Parameters
        ----------
        pred: Dict[int, np.array]
            predictions of model, aggregated in dict by user id
        target: Dict[int, np.array]
            target items, aggregated in dict by user id

        Returns
        -------
        float:
            value of the metric.
        """
        if set(pred.keys()) != set(target.keys()):
            raise ValueError("User ids in predictions is not equals to user ids in target.")

        sum_metric = 0
        for user_id in pred.keys():
            sum_metric += self._apk(pred=pred[user_id], target=target[user_id])
        return sum_metric / len(pred.keys())
