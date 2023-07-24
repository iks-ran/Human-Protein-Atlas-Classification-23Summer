from torch import nn
import torch
import os
from torch.utils.data import DataLoader
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from .metrics import logits2label, F1
import pandas as pd

class Trainer(nn.Module):
    def __init__(self, model, optimizer, loss_func, train_dataset, val_dataset, hparams):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.hparams = hparams

    def _init_train(self):

        self.start = 1
        self.tmp = self.start
        self.end = self.hparams.train.epochs
        self.lr_warmup = np.linspace(0, self.hparams.train.optimizer.lr, self.hparams.train.warmup)
        self.schedular = CosineAnnealingLR(self.optimizer, T_max=self.end - self.hparams.train.warmup)

        if self.hparams.train.device == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.hparams.train.device)
        self.model.to(self.device)

        if not self.hparams.resume == '':
            state_dict = torch.load(self.hparams.resume)
            self.model.load_state_dict(state_dict['checkpoint'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.schedular.load_state_dict(state_dict['schedular'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device=self.device, non_blocking=True)
            for state in self.schedular.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device=self.device, non_blocking=True)
            self.start = state_dict['epoch']
            self.tmp = self.start

        self.train_loader = DataLoader(self.train_dataset, **self.hparams.train.dataloader)
        self.batch_size = self.train_loader.batch_size
        self.iters = len(self.train_dataset) // self.batch_size

        base_path = os.path.dirname(__file__)
        exp_path = f'../../exp/{self.hparams.model.name}/{datetime.now().strftime("%m-%d-%H-%M")}'
        self.log_path = os.path.join(base_path, exp_path)
        os.makedirs(self.log_path, exist_ok=True)

        self.best_loss = 1e10
        self.start_time = datetime.now().replace(microsecond=0)

    def _init_val(self):

        if isinstance(self.hparams.train.internal, list):
            self.val_internal = self.hparams.train.internal
        else:
            self.val_internal = [i for i in range(self.start, self.end + 1) if i % self.hparams.train.internal == 0]
        self.val_loader = DataLoader(self.val_dataset, **self.hparams.val.dataloader)
        self.val_samples = len(self.val_dataset)
        os.makedirs(f'{self.log_path}/predict', exist_ok=True)
        val_labels = []
        for batch in self.val_loader:
            val_labels.append(batch['label'].numpy())
        self.val_labels = np.concatenate(val_labels, axis=0)
        self.best_f1 = 0

    def run(self):
        
        self._init_train()
        self._init_val()

        for self.tmp in range(self.start, self.end + 1):

            if self.tmp <= self.hparams.train.warmup:
                self.warmup(self.tmp)
            else:
                self.schedular.step()

            train_loss, train_log = self.run_epoch(self.tmp)
            val_log = ''
            if self.tmp in self.val_internal:
                val_loss, val_f1, val_log = self.evaluate()
                if val_loss < self.best_loss:
                    self.save_model('model_loss_best')
                    self.best_loss = val_loss
                if val_f1 > self.best_f1:
                    self.save_model('model_f1_best')
                    self.best_f1 = val_f1

            with open(f'{self.log_path}/log.log', 'a') as f:
                f.write(train_log + val_log)

            self.save_model('model_last')
        
    def run_epoch(self, epoch):
    
        self.model.train()
        epoch_loss = 0
        epoch_log = ''
        
        for i, batch in enumerate(self.train_loader, start=1):
            if i > self.iters:
                break
            inputs = batch['inp'].to(self.device)
            labels = batch['label'].to(self.device)
            outputs = self.model(inputs)
            iter_loss = self.loss_func(outputs, labels)
            iter_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_per_sample = iter_loss / self.batch_size
            epoch_loss += loss_per_sample
            time_elapsed = datetime.now() - self.start_time
            time_reserved = time_elapsed / ((epoch - 1) * self.iters + i) * self.end * self.iters
            msg = f'Time: [ {time_elapsed} / {time_reserved} ] Iteration: [ {i} / {self.iters} ] ' + \
                  f'Epoch: [ {epoch} / {self.end} ] Loss: {loss_per_sample:.4f} '
            epoch_log += (msg + '\n')
            print(msg)
            
        epoch_loss /= self.iters

        msg = f'Epoch: [ {epoch} / {self.end} ] Loss: {epoch_loss:.4f} Best Validation Loss {self.best_loss:.4f}'
        epoch_log += (msg + '\n')
        print(msg)
        
        return epoch_loss, epoch_log
    
    def evaluate(self):
        
        self.model.eval()
        val_loss = 0
        val_log = ''
        pred = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                inputs = batch['inp'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(inputs)
                pred.append(torch.sigmoid(outputs).cpu().numpy())
                val_loss += self.loss_func(outputs, labels)

        val_loss /= self.val_samples
        pred = np.concatenate(pred, axis=0)
        f1 = F1(self.val_labels, logits2label(pred))
        PRED = pd.DataFrame(pred)
        PRED.to_csv(f'{self.log_path}/predict/epoch_{self.tmp}.csv', index=False)

        msg = f'Epoch [ {self.tmp} / {self.end} ] Validation Loss: {val_loss:.4f} Best Validation Loss: {self.best_loss:.4f} F1: {f1:.4f} Best F1: {self.best_f1:.4f}'
        val_log += (msg + '\n')
        print(msg)
        
        return val_loss, f1, val_log

    def warmup(self, epoch):
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr_warmup[epoch - 1]

    def save_model(self, model_name):
        checkpoint = dict(model=self.model.state_dict(), 
                          epoch=self.tmp, 
                          optimizer=self.optimizer.state_dict(), 
                          schrdular=self.schedular.state_dict())
        torch.save(checkpoint, f'{self.log_path}/{model_name}.pth')
