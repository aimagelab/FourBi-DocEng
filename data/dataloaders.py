import torch
from torch.utils.data import Dataset

from utils.htr_logging import get_logger

logger = get_logger(__file__)


def make_test_dataloader(test_dataset: Dataset, config: dict):
    test_dataloader_config = config['test_kwargs']
    test_data_loader = torch.utils.data.DataLoader(test_dataset, **test_dataloader_config)

    return test_data_loader
