import os
import time
import pprint
import argparse
import torch
import numpy as np
import pickle
import utils
import csv

from model.hidden import Hidden
from noise_layers.noiser import Noiser
from average_meter import AverageMeter

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class SingleDirectoryDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB to handle grayscale images
        if self.transform:
            image = self.transform(image)
        filename = os.path.basename(img_path)
        return image, filename


def write_validation_loss(file_name, losses_accu, experiment_name, epoch, write_header=False):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            row_to_write = ['experiment_name', 'epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()]
            writer.writerow(row_to_write)
        row_to_write = [experiment_name, epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()]
        writer.writerow(row_to_write)


def main():
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    # parser.add_argument('--size', '-s', default=128, type=int, help='The size of the images (images are square so this is height and width).')
    parser.add_argument('--data-dir', '-d', required=True, type=str, help='The directory where the data is stored.')
    parser.add_argument('--runs_root', '-r', default=os.path.join('.', 'experiments'), type=str,
                        help='The root folder where data about experiments are stored.')
    parser.add_argument('--batch-size', '-b', default=1, type=int, help='Validation batch size.')
    parser.add_argument('--ouput-dir', '-o', default="/home/lai/Research/Graduate/HiDDeN/output", type=str, help='The directory where the output will be stored.')

    args = parser.parse_args()
    print_each = 25

    completed_runs = [o for o in os.listdir(args.runs_root)
                      if os.path.isdir(os.path.join(args.runs_root, o)) and o != 'no-noise-defaults']

    print(completed_runs)

    write_csv_header = True
    for run_name in completed_runs:
        current_run = os.path.join(args.runs_root, run_name)
        print(f'Run folder: {current_run}')
        options_file = os.path.join(current_run, 'options-and-config.pickle')
        train_options, hidden_config, noise_config = utils.load_options(options_file)
        train_options.train_folder = os.path.join(args.data_dir)
        train_options.validation_folder = os.path.join(args.data_dir)
        train_options.batch_size = args.batch_size
        checkpoint, chpt_file_name = utils.load_last_checkpoint(os.path.join(current_run, 'checkpoints'))
        print(f'Loaded checkpoint from file {chpt_file_name}')

        noiser = Noiser(noise_config, device=device)
        model = Hidden(hidden_config, device, noiser, tb_logger=None)
        utils.model_from_checkpoint(model, checkpoint)

        print('Model loaded successfully. Starting validation run...')

        test_transform = transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Create the dataset
        test_dataset = SingleDirectoryDataset(directory=args.data_dir, transform=test_transform)

        # Create the DataLoader
        test_dataloader = DataLoader(test_dataset, batch_size=train_options.batch_size, shuffle=False, num_workers=4)

        file_count = len(test_dataloader.dataset)
        print(f'Found {file_count} files in validation dataset')
        if file_count % train_options.batch_size == 0:
            steps_in_epoch = file_count // train_options.batch_size
        else:
            steps_in_epoch = file_count // train_options.batch_size + 1

        losses_accu = {}
        step = 0
        for image, filename in test_dataloader:
            step += 1
            print(f'Step {step}/{steps_in_epoch}')
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([image, message])
            # print(encoded_images.shape, noised_images.shape, decoded_messages.shape)
            # print(encoded_images.min(), encoded_images.max())
            # save images
            for i in range(image.shape[0]):
                print("Saving images for {} to {}".format(filename[i], args.ouput_dir))
                img = image[i].detach().cpu().numpy().transpose(1, 2, 0)
                img = (img + 1) / 2
                img = (img * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(args.ouput_dir, f'{filename[i]}_original.png'))

                img = encoded_images[i].detach().cpu().numpy().transpose(1, 2, 0)
                img = (img + 1) / 2
                img = (img * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(args.ouput_dir, f'{filename[i]}_encoded.png'))

                img = noised_images[i].detach().cpu().numpy().transpose(1, 2, 0)
                img = (img + 1) / 2
                img = (img * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(args.ouput_dir, f'{filename[i]}_noised.png'))

            if not losses_accu:  # dict is empty, initialize
                for name in losses:
                    losses_accu[name] = AverageMeter()
            for name, loss in losses.items():
                losses_accu[name].update(loss)
            if step % print_each == 0 or step == steps_in_epoch:
                print(f'Step {step}/{steps_in_epoch}')
                utils.print_progress(losses_accu)
                print('-' * 40)

        # utils.print_progress(losses_accu)
        write_validation_loss(os.path.join(args.runs_root, 'validation_run.csv'), losses_accu, run_name,
                              checkpoint['epoch'],
                              write_header=write_csv_header)
        write_csv_header = False

    # train(model, device, hidden_config, train_options, this_run_folder, tb_logger)


if __name__ == '__main__':
    main()