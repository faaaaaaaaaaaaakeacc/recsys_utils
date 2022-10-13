import numpy as np
from typing import Tuple


class RecSysDatasetBase:
    """Base class for recommendation dataset."""

    def get_ui_matrix(self) -> Tuple[np.array, np.array, np.array]:
        """Computes user-item matrix.

        Returns
        -------
        Tuple:
            np.array: user-item matrix.
            np.array: ids of users.
            np.array: ids of items.
        """
        pass
