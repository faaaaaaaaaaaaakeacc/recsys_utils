import numpy as np
from typing import Dict


class BaseMetric:
    """Base class for metric computation."""

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
        pass
