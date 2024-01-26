import json, argparse, time, os, shutil, itertools
import pandas as pd
import numpy as np
from os.path import join, exists
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import *
from data_loader import Yield, Weather
from networks import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")


class Trainer(object):
    def __init__(self, trial_dir: str, config: argparse.Namespace):
        self.trial_dir = trial_dir
        self.config = config
        self.writer = SummaryWriter(join(self.trial_dir, f'tensorboard_r{self.config.repeat_num}'))

        # Set device on GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def setup_model(self):
        """ 
        Set up model and loss function.
        """
        
        G_theta = Generator(self.config.latent_dim).to(self.device)
        D_theta = Discriminator(self.config.latent_dim).to(self.device)

        criterion_GAN = nn.MSELoss().to(self.device)
   
          
        return (G_theta, D_theta, criterion_GAN)

    def setup_optimizer(self, G_theta, D_theta):
        """ 
        Set up optimizer for network
        """
        
        optimizer_G = optim.Adam(
                    G_theta.parameters(),  # Assuming G_theta is a model with learnable parameters
                    lr=self.config.learning_rate,
                    betas=(self.config.beta1, 0.999),
                    weight_decay=self.config.weight_decay,
                    )

        optimizer_D = optim.Adam(
                    D_theta.parameters(),  # Assuming D_theta is a model with learnable parameters
                    lr=self.config.learning_rate,
                    betas=(self.config.beta1, 0.999),
                    weight_decay=self.config.weight_decay,
                    )
        return (optimizer_G, optimizer_D)


    def train(self):  
        X_train_valid = pd.read_csv(f'{self.config.data_path}/train_data.csv')
        noise_full = np.load(f'{self.config.data_path}/noise.npy')

        train_subs, valid_subs = train_test_split(
            X_train_valid, test_size=0.2, random_state=self.config.random_seed,
        )
        noise_train_subs, noise_valid_subs = train_test_split(
            noise_full, test_size=0.2, random_state=self.config.random_seed,
        )
        
        X_train = train_subs
        X_valid = valid_subs
        
        N_train = noise_train_subs
        N_valid = noise_valid_subs

        num_train = len(X_train)
        
        yield_train_loader = DataLoader(
            dataset=Yield(data=X_train), 
            batch_size=self.config.batch_size, num_workers=self.config.num_workers, 
            shuffle=True, pin_memory=True, drop_last=(len(X_train)%self.config.batch_size == 1), 
        )

        noise_train_loader = DataLoader(
            dataset=Weather(data=N_train), 
            batch_size=self.config.batch_size, num_workers=self.config.num_workers, 
            shuffle=True, pin_memory=True, drop_last=(len(X_train)%self.config.batch_size == 1), 
        )


        yield_valid_loader = DataLoader(
            dataset=Yield(data=X_valid), 
            batch_size=self.config.batch_size, num_workers=self.config.num_workers, 
            shuffle=True, pin_memory=True, drop_last=(len(X_train)%self.config.batch_size == 1), 
        )

        noise_valid_loader = DataLoader(
            dataset=Weather(data=N_valid), 
            batch_size=self.config.batch_size, num_workers=self.config.num_workers, 
            shuffle=True, pin_memory=True, drop_last=(len(X_train)%self.config.batch_size == 1), 
        )
        self.G_theta, self.D_theta, self.criterion_GAN = self.setup_model()
        self.optimizer_G, self.optimizer_D = self.setup_optimizer(self.G_theta, self.D_theta)

        self.start_epoch = 1 
        self.best_valid_perf = None
        self.counter = 0

        for epoch in range(self.start_epoch, self.config.epochs + 1):

            train_loss, train_time = self.train_one_epoch(epoch, yield_train_loader, noise_train_loader, num_train)
            valid_loss = self.validate(epoch, yield_valid_loader, noise_valid_loader)

            # Check for improvement
            if self.best_valid_perf is None:
                self.best_valid_perf = valid_loss
                is_best = True
            else:
                is_best = valid_loss < self.best_valid_perf          
            msg = "Epoch:{}, {:.1f}s - train loss: {:.6f} - validation loss: {:.6f}"
            if is_best:
                msg += " [*]"
                self.counter = 0
            print(msg.format(epoch, train_time, train_loss, valid_loss))

            # Checkpoint the model
            if not is_best:
                self.counter += 1
            if self.counter > self.config.train_patience and self.config.early_stop:
                print("[!] No improvement in a while, stopping training.")
                break
            self.best_valid_perf = min(valid_loss, self.best_valid_perf)
            self.save_checkpoint(
            {
                'epoch': epoch,
                'G_theta_state': self.G_theta.state_dict(),
                'D_theta_state': self.D_theta.state_dict(),
                'optimizer_G_state': self.optimizer_G.state_dict(), 
                'optimizer_D_state': self.optimizer_D.state_dict(), 
                'best_valid_perf': self.best_valid_perf,
                'last_valid_perf': valid_loss,
                'counter': self.counter,
            }, repeat_num=self.config.repeat_num, split_num=self.config.split_num, is_best=is_best)
                
            self.writer.flush()
            self.writer.close()

        print("\ndone!")

    def train_one_epoch(self, epoch, yield_train_loader, noise_train_loader, num_train):
        batch_time = AverageMeter()
        batch_time.reset()
        losses = AverageMeter()
        losses.reset()
        tic = time.time()

        self.G_theta.train()
        self.D_theta.train()

        with tqdm(total=num_train) as pbar:  
            for batch_index, (real_yield, input_noise) in enumerate(zip(yield_train_loader,noise_train_loader)): 

                batch_size = real_yield.shape[0]
                real_yield = real_yield.to(self.device)
                noise = input_noise

                # Train Generator
                self.optimizer_G.zero_grad()

                fake_yield = self.G_theta(noise)

                if batch_index % 10 == 0:

                    real_yield_mean = torch.mean(real_yield, dim=0).detach().cpu().numpy()
                    fake_yield_mean = torch.mean(fake_yield, dim=0).detach().cpu().numpy()
                    tag = 'Train/Generated_Points_Distribution'
                    flattened_points = fake_yield.view(-1)
                    self.writer.add_histogram(tag, flattened_points, global_step=batch_index + (epoch-1))

                loss_yield_fake = self.criterion_GAN(self.D_theta(fake_yield), torch.ones_like(self.D_theta(fake_yield)))
                loss_G = loss_yield_fake
                loss_G.backward()
                self.optimizer_G.step()

                # Train Discriminator
                self.optimizer_D.zero_grad()

                loss_D_yield_real = self.criterion_GAN(self.D_theta(real_yield), torch.ones_like(self.D_theta(real_yield)))
                loss_D_yield_fake = self.criterion_GAN(self.D_theta(fake_yield.detach()), torch.zeros_like(self.D_theta(fake_yield)))
                loss_D_yield = (loss_D_yield_real + loss_D_yield_fake) * 0.5
                loss_D_yield.backward()

                loss_D = loss_D_yield 
                self.optimizer_D.step()

                loss = (loss_G + loss_D) * 0.5
                                # update metric
                losses.update(loss.item(), batch_size)
                
                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)
                tic = time.time()
                
                pbar.set_description(("{:.1f}s - loss: {:.3f}".format(batch_time.val, losses.val)))
                pbar.update(batch_size)
                
                self.writer.add_scalar('Train/loss_G', loss_D.item(), global_step=batch_index + (epoch-1))
                self.writer.add_scalar('Train/loss_D_yield_real', loss_D_yield_real.item(), global_step=batch_index + (epoch-1))
                self.writer.add_scalar('Train/loss_D_yield_fake', loss_D_yield_fake.item(), global_step=batch_index + (epoch-1))
                self.writer.add_scalar('Train/loss_D', loss_D.item(), global_step=batch_index + (epoch-1))
                self.writer.add_scalar('Train/loss_total', loss.item(), global_step=batch_index + (epoch-1))
             
        # Write in tensorboard
        self.writer.add_scalar('Loss/train', losses.avg, epoch)

        return losses.avg, batch_time.sum
        
    def validate(self, epoch, yield_valid_loader, noise_valid_loader):
        losses = AverageMeter()
        losses.reset()
        
        # switch to evaluate mode
        self.G_theta.eval()
        self.D_theta.eval()

        
        with torch.no_grad():
            for batch_index, (real_yield, input_noise) in enumerate(zip(yield_valid_loader, noise_valid_loader)): 

                batch_size = real_yield.shape[0]
                real_yield = real_yield.to(self.device)
                noise = input_noise

                fake_yield = self.G_theta(noise)

                if batch_index % 10 == 0:

                    real_yield_mean = torch.mean(real_yield, dim=0).detach().cpu().numpy()
                    fake_yield_mean = torch.mean(fake_yield, dim=0).detach().cpu().numpy()
                    tag = 'Valid/Generated_Points_Distribution'
                    flattened_points = fake_yield.view(-1)
                    self.writer.add_histogram(tag, flattened_points, global_step=batch_index + (epoch-1))

               
                loss_yield_fake = self.criterion_GAN(self.D_theta(fake_yield), torch.ones_like(self.D_theta(fake_yield)))
                loss_G = loss_yield_fake

                loss_D_yield_real = self.criterion_GAN(self.D_theta(real_yield), torch.ones_like(self.D_theta(real_yield)))
                loss_D_yield_fake = self.criterion_GAN(self.D_theta(fake_yield.detach()), torch.zeros_like(self.D_theta(fake_yield)))
                loss_D_yield = (loss_D_yield_real + loss_D_yield_fake) * 0.5
                loss_D = loss_D_yield 

                loss = (loss_G + loss_D) * 0.5
                # update metric
                losses.update(loss.item(), batch_size)


            # Write in tensorboard
            self.writer.add_scalar('Valid/loss', losses.avg, epoch)

        return losses.avg

        

    def test(self): 
        pass



    def save_checkpoint(self, state, repeat_num, split_num, is_best):            
        filename = f'model_ckpt_r{repeat_num}s{split_num}.tar'
        ckpt_path = join(self.trial_dir, filename)
        torch.save(state, ckpt_path)
        
        if is_best:
            filename = f'best_model_ckpt_r{repeat_num}s{split_num}.tar'
            shutil.copyfile(ckpt_path, join(self.trial_dir, filename))





        