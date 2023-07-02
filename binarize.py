import argparse
import torch
from pathlib import Path
from trainer.FourBiInference import FourbiInferenceModule
from data.TestDataset import FolderDataset
from torchvision import transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binarize a folder of images')
    parser.add_argument('model', type=str, metavar='PATH', help='path to the model file')
    parser.add_argument('--src', type=str, required=True, help='path to the folder of input images')
    parser.add_argument('--dst', type=str, default='output', help='path to the folder of output images')
    parser.add_argument('--device', type=str, default='cude', help='device to use')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = FourbiInferenceModule(config={'resume': args.model, 'batch_size': args.batch_size}, device=device)

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    assert src.is_dir(), f'{src} is not a directory'

    patch_size = 512
    dataset = FolderDataset(src, patch_size=patch_size, overlap=True, transform=transforms.ToTensor())
    model.test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model.config['test_patch_size'] = patch_size
    model.config['test_stride'] = patch_size // 2

    for i, sample in enumerate(model.folder_test()):
        key = list(sample.keys())[0]
        img, pred, gt = sample[key]
        src_img_path = Path(key)

        dst_img_path = dst / (src_img_path.stem + '.png')
        pred.save(str(dst_img_path))
        print(f'({i + 1}/{len(dataset)}) Saving {dst_img_path}')
    print('Done.')

