from typing import List
# from ..model.padded_tranformer import PaddedTransformer
from datasets import Dataset

class Task():
    def __init__(
        self,
        padded_transformer: PaddedTransformer,
        dataset: Dataset,
        task: str,
        metrics: List[str]
    ):
    pass
