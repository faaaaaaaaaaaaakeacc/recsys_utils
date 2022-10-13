import pandas as pd
import numpy as np
from typing import Optional
from typing import Tuple
from base import RecSysDatasetBase
from tqdm.auto import trange


class RecSysSequentialDataset(RecSysDatasetBase):
    """Class for holding sequential dataset."""

    def __init__(self,
                 df: pd.DataFrame,
                 users_column: str,
                 items_column: str,
                 marks_column: Optional[str],
                 use_tqdm: bool = True,
                 ):
        """Init RecSysSequentialDataset.

        Parameters
        ----------
        df: pd.DataFrame
            dataset users transactions.
        users_column: str
            name of column with user's ids.
        items_column: str
            name of column with item's ids.
        marks_column: Optional[str]
            name of column with user's marks for items.

        """
        self.df = df
        self.users_column = users_column
        self.items_column = items_column
        self.marks_column = marks_column
        self.use_tqdm = use_tqdm
        self.users_id = np.array(df[users_column].unique())
        self.items_id = np.array(df[items_column].unique())

    def get_ui_matrix(self) -> Tuple[np.array, np.array, np.array]:
        """Computes user-item matrix.

        Returns
        -------
        Tuple:
            np.array: user-item matrix.
            np.array: ids of users.
            np.array: ids of items.
        """
        iter = range if self.use_tqdm else trange
        answer = np.zeros((len(self.users_id), len(self.items_id)))
        for i in iter(self.df.shape[0]):
            cur_user = self.df[self.users_column].iloc[i]
            cur_item = self.df[self.items_column].iloc[i]
            cur_mark = 1
            if self.marks_column:
                cur_mark = self.df[self.marks_column].iloc[i]
            answer[cur_user][cur_item] = cur_mark
        return answer, self.users_id, self.items_id

