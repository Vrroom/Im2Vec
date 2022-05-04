import itertools
import math
import os
import random
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from scheduler import GradualWarmupScheduler
import matplotlib.pyplot as plt
import gc
from PIL import Image
import glob
from Ranger import Ranger
import torch_optimizer as optim_

class MyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = glob.glob(image_paths+ '/train/*.png')
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform:
            x = self.transform(x)

        return x, ''

    def __len__(self):
        return len(self.image_paths)

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.first_epoch = True
        self.beta_scale = 2.0  #1.4
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device
        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.logger.experiment.log({'loss': train_loss['loss']})
        max_paths = 25
        if self.model.only_auxillary_training:
            path = self.current_epoch + 6
            if path>30:
                path = random.randint(7, 25)
                self.model.save_lossvspath = False
        else:
            path = random.randint(7, 25)
        if self.params['grow']:
            self.model.redo_features(path)
        return train_loss

    def on_load_checkpoint(self, checkpoint):
        load_epoch = checkpoint['epoch']
        new_beta = self.model.beta * (self.beta_scale ** (load_epoch // 25))
        self.model.beta = min(new_beta, 4)
        print('loaded beta: ', self.model.beta)

    def training_epoch_end(self, outputs):
        super(VAEXperiment, self).training_epoch_end(outputs)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss, 'learning_rate': self.trainer.optimizers[0].param_groups[0]["lr"]}
        self.sample_images()
        if (self.current_epoch+1) % self.model.memory_leak_epochs == 0 and self.model.memory_leak_training and not self.first_epoch:
            quit()
        self.first_epoch = False
        print('beta: ', self.model.beta)
        if self.current_epoch % 25 ==0:
            new_beta = self.model.beta * self.beta_scale
            self.model.beta = min(new_beta, 4)
        gc.collect()
        torch.cuda.empty_cache()
        print('learning rate: ', self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log('val_loss', avg_loss)

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch:04d}.png",
                          normalize=False,
                          nrow=12)

        vutils.save_image(test_input.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"real_img_{self.logger.name}_{self.current_epoch:04d}.png",
                          normalize=False,
                          nrow=12)

        del test_input, recons #, samples

    def sample_interpolate(self, save_dir, name, version, save_svg=False, other_interpolations=False):
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        interpolate_samples = self.model.interpolate(test_input, verbose=False)
        interpolate_samples = torch.cat(interpolate_samples, dim=0)
        vutils.save_image(interpolate_samples.cpu().data,
                          f"{save_dir}{name}/version_{version}/"
                          f"{name}_interpolate_img.png",
                          normalize=False,
                          nrow=10)

        if other_interpolations:
            interpolate_samples = self.model.interpolate2D(test_input, verbose=False)
            interpolate_samples = torch.cat(interpolate_samples, dim=0)
            vutils.save_image(interpolate_samples.cpu().data,
                              f"{save_dir}{name}/version_{version}/"
                              f"{name}_interpolate2D_image.png",
                              normalize=False,
                              nrow=10)
            interpolate_samples = self.model.interpolate2D(test_input, verbose=True)
            interpolate_samples = torch.cat(interpolate_samples, dim=0)
            vutils.save_image(interpolate_samples.cpu().data,
                              f"{save_dir}{name}/version_{version}/"
                              f"{name}_interpolate2D_vector.png",
                              normalize=False,
                              nrow=10)
            interpolate_samples = self.model.visualize_sampling(test_input, verbose=False)
            interpolate_samples = torch.cat(interpolate_samples, dim=0)
            vutils.save_image(interpolate_samples.cpu().data,
                              f"{save_dir}{name}/version_{version}/"
                              f"{name}_visualize_sampling_image.png",
                              normalize=False,
                              nrow=self.params['val_batch_size'])
            interpolate_samples = self.model.visualize_sampling(test_input, verbose=True)
            interpolate_samples = torch.cat(interpolate_samples, dim=0)
            vutils.save_image(interpolate_samples.cpu().data,
                              f"{save_dir}{name}/version_{version}/"
                              f"{name}_visualize_sampling_vector.png",
                              normalize=False,
                              nrow=self.params['val_batch_size'])
            interpolate_samples = self.model.naive_vector_interpolate(test_input, verbose=False)
            interpolate_samples = torch.cat(interpolate_samples, dim=0)
            vutils.save_image(interpolate_samples.cpu().data,
                              f"{save_dir}{name}/version_{version}/"
                              f"{name}_naive_interpolate_image.png",
                              normalize=False,
                              nrow=10)
            interpolate_samples = self.model.naive_vector_interpolate(test_input, verbose=True)
            interpolate_samples = torch.cat(interpolate_samples, dim=0)
            vutils.save_image(interpolate_samples.cpu().data,
                              f"{save_dir}{name}/version_{version}/"
                              f"{name}_naive_interpolate_vector.png",
                              normalize=False,
                              nrow=10)
            interpolate_samples = self.model.interpolate(test_input, verbose=True)
            interpolate_samples = torch.cat(interpolate_samples, dim=0)
            vutils.save_image(interpolate_samples.cpu().data,
                              f"{save_dir}{name}/version_{version}/"
                              f"{name}_interpolate_vector.png",
                              normalize=False,
                              nrow=10)
            sampling_graph = self.model.sampling_error(test_input)
            plt.imsave(f"{save_dir}{name}/version_{version}/{name}_recons_graph.png", sampling_graph)
            if self.model.only_auxillary_training:
                graph = self.model.visualize_aux_error(test_input, verbose=True)
                plt.imsave(f"{save_dir}{name}/version_{version}/{name}_aux_graph.png", graph)

        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.cpu().data,
                          f"{save_dir}{name}/version_{version}/"
                          f"{name}_recons.png",
                          normalize=False,
                          nrow=10)
        vutils.save_image(test_input.cpu().data,
                          f"{save_dir}{name}/version_{version}/"
                          f"{name}_input.png",
                          normalize=False,
                          nrow=10)
        # if save_svg:
        #     self.model.save(test_input, save_dir, name)

    def configure_optimizers(self):
        optims = []
        scheds = []
        if self.model.only_auxillary_training:
            print('Learning Rate changed for auxillary training')
            self.params['LR'] = 0.00001
        optimizer = Ranger(self.model.parameters(),
                                   lr=self.params['LR'],
                                   weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optims[0], 'min', verbose=True, factor=self.params['scheduler_gamma'], min_lr=0.0001, patience=int(self.model.memory_leak_epochs/7))
        scheds.append({
         'scheduler': scheduler,
         'monitor': 'val_loss', # Default: val_loss
         'interval': 'epoch',
         'frequency': 1,
        },)
        print('USING WARMUP SCHEDULER')
        return optims, scheds

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root = self.params['data_path'],
                             split = "train",
                             transform=transform,
                             download=False)
        elif self.params['dataset'] == 'MNIST':
            dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 64,
                                                 shuffle = False,
                                                 drop_last=True)
            self.num_val_imgs = 200
            in_channels = 1
        else:
            dataset = datasets.ImageFolder(self.params['data_path'], transform=transform)
            # dataset = MyDataset(self.params['data_path'], transform=transform)
            test_dataset = self.params['data_path'].replace('train','test')
            if os.path.exists(test_dataset):
                test_dataset = datasets.ImageFolder(test_dataset, transform=transform)
            else:
                test_dataset = dataset
            self.sample_dataloader =  DataLoader(test_dataset,
                                                 batch_size= self.params['val_batch_size'],
                                                 shuffle = self.params['val_shuffle'],
                                                 drop_last=True, num_workers=1)
            self.num_val_imgs = len(self.sample_dataloader)

            # raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          shuffle = True,
                          drop_last=False, num_workers=1)

    # @data_loader
    # def val_dataloader(self):
    #     transform = self.data_transforms()
    #
    #     if self.params['dataset'] == 'celeba':
    #         self.sample_dataloader =  DataLoader(CelebA(root = self.params['data_path'],
    #                                                     split = "test",
    #                                                     transform=transform,
    #                                                     download=False),
    #                                              batch_size= 144,
    #                                              shuffle = True,
    #                                              drop_last=True)
    #         self.num_val_imgs = len(self.sample_dataloader)
    #     else:
    #         dataset = datasets.ImageFolder(self.params['data_path'], transform=transform)
    #         self.sample_dataloader =  DataLoader(dataset,
    #                                              batch_size= 64,
    #                                              shuffle = False,
    #                                              drop_last=True)
    #         self.num_val_imgs = 200#len(self.sample_dataloader)
    #
    #     return self.sample_dataloader
    #
    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: (2 * X - 1.))
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            transform = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                            transforms.Resize(self.params['img_size']),
                                            # transforms.RandomRotation([0, 360], resample=3, fill=(255,255,255)),
                                            # transforms.RandomAffine([0, 0], (0.0,0.05), (1.0,1.0), resample=3, fillcolor=(255,255,255)),
                                            transforms.CenterCrop(self.params['img_size']),
                                            transforms.ToTensor(),
                                            ])
            # raise ValueError('Undefined dataset type')
        return transform
