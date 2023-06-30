import math
import random

import numpy as np
import torch
import torch.utils.data
from torchvision.transforms import functional

from data.dataloaders import make_test_dataloader
from data.datasets import make_test_dataset
from data.utils import reconstruct_ground_truth
from modules.FFC import Fourbi
from utils.htr_logging import get_logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calculate_psnr(predicted: torch.Tensor, ground_truth: torch.Tensor, threshold=0.5):
    pred_img = predicted.detach().cpu().numpy()
    gt_img = ground_truth.detach().cpu().numpy()

    pred_img = (pred_img > threshold) * 1.0

    mse = np.mean((pred_img - gt_img) ** 2)
    psnr = 100 if mse == 0 else (20 * math.log10(1.0 / math.sqrt(mse)))
    return psnr


class FourbiInferenceModule:

    def __init__(self, config, device=None):

        self.config = config
        self.device = device
        self.checkpoint = None
        self.batch_size = config['batch_size']

        if 'resume' in self.config:
            self.checkpoint = torch.load(config['resume'], map_location=device)
            checkpoint_config = self.checkpoint['config'] if 'config' in self.checkpoint else {}
            self.config.update(checkpoint_config)
            config = self.config

        self.model = Fourbi(input_nc=config['input_channels'], output_nc=config['output_channels'],
                            n_downsampling=config['n_downsampling'], init_conv_kwargs=config['init_conv_kwargs'],
                            downsample_conv_kwargs=config['down_sample_conv_kwargs'],
                            resnet_conv_kwargs=config['resnet_conv_kwargs'], n_blocks=config['n_blocks'],
                            use_convolutions=config['use_convolutions'],
                            skip_connections=config['skip_connections'],
                            unet_layers=config['unet_layers'], )

        config['num_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.seed = config['seed']

        if self.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint['model'], strict=True)
            self.load_random_settings()

        self.model = self.model.to(self.device)

        # Logging
        self.logger = get_logger(FourbiInferenceModule.__name__)

    def load_random_settings(self):
        if 'random_settings' in self.checkpoint:
            set_seed(self.checkpoint['random_settings']['seed'])
            random.setstate(self.checkpoint['random_settings']['random_rng_state'])
            np.random.set_state(self.checkpoint['random_settings']['numpy_rng_state'])
            torch.set_rng_state(self.checkpoint['random_settings']['torch_rng_state'].type(torch.ByteTensor))
            torch.cuda.set_rng_state(self.checkpoint['random_settings']['cuda_rng_state'].type(torch.ByteTensor))

    def eval_item(self, item, threshold):
        image_name = item['image_name'][0]
        sample = item['sample']
        num_rows = item['num_rows'].item()
        samples_patches = item['samples_patches']
        gt_sample = item['gt_sample']

        samples_patches = samples_patches.squeeze(0)
        test = samples_patches.to(self.device)
        gt_test = gt_sample.to(self.device)

        test = test.squeeze(0)
        test = test.permute(1, 0, 2, 3)

        pred = []
        for chunk in torch.split(test, self.batch_size):
            pred.append(self.model(chunk))
        pred = torch.cat(pred)

        pred = reconstruct_ground_truth(pred, gt_test, num_rows=num_rows, config=self.config)

        pred = torch.where(pred > threshold, 1., 0.)

        test = sample.squeeze(0).detach()
        pred = pred.squeeze(0).detach()
        gt_test = gt_test.squeeze(0).detach()
        test_img = functional.to_pil_image(test)
        pred_img = functional.to_pil_image(pred)
        gt_test_img = functional.to_pil_image(gt_test)
        images = {image_name: [test_img, pred_img, gt_test_img]}

        return images

    @torch.no_grad()
    def folder_test(self):
        self.model.eval()
        threshold = self.config['threshold']

        for i, item in enumerate(self.test_data_loader):
            images_item = self.eval_item(item, threshold)
            yield images_item
