from torchvision.transforms import transforms
import time
from data.TestDataset import TestDataset
from utils.htr_logging import get_logger
from torch.utils.data import ConcatDataset

logger = get_logger(__file__)


def make_test_dataset(config: dict, is_validation=False):
    test_data_path = config['test_data_path']
    patch_size = config['test_patch_size']
    stride = config['test_stride']
    load_data = config['load_data']

    transform = transforms.Compose([transforms.ToTensor()])

    logger.info(f"Loading test datasets...")
    time_start = time.time()
    datasets = []
    for path in test_data_path:
        datasets.append(
            TestDataset(
                data_path=path,
                patch_size=patch_size,
                stride=stride,
                transform=transform,
                is_validation=is_validation,
                load_data=load_data))
        logger.info(f'Loaded test dataset from {path} with {len(datasets[-1])} instances.')

    logger.info(f"Loading test datasets took {time.time() - time_start:.2f} seconds")

    test_dataset = ConcatDataset(datasets)

    logger.info(f"Test set has {len(test_dataset)} instances")
    return test_dataset
