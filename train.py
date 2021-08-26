import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import model
from tqdm import tqdm
import transformers
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import mean_squared_error

import numpy as np


class Config:
    NB_EPOCHS = 10
    LR = 2e-5
    MAX_LEN = 256
    N_SPLITS = 5
    TRAIN_BS = 16
    VALID_BS = 32
    FILE_NAME = '../input/commonlitreadabilityprize/train.csv'
    TOKENIZER = transformers.AutoTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    scaler = GradScaler()


class dataset(Dataset):
    def __init__(self):
        super(dataset, self).__init__()

    def __len__(self):
        pass
    def __getitem__(self, x):
        pass


m = model(Config,"bert-base-chinese","cls")


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            scheduler,
            train_dataloader,
            valid_dataloader,
            device
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.loss_fn = self.yield_loss
        self.device = device

    def yield_loss(self, outputs, targets):
        """
        This is the loss function for this task
        """
        loss = nn.MSELoss()
        return torch.sqrt(loss(outputs, targets))

    def train_one_epoch(self):
        """
        This function trains the model for 1 epoch through all batches
        """
        prog_bar = tqdm(enumerate(self.train_data), total=len(self.train_data))
        self.model.train()
        with autocast():
            for idx, inputs in prog_bar:
                ids = inputs['ids'].to(self.device, dtype=torch.long)
                mask = inputs['mask'].to(self.device, dtype=torch.long)
                targets = inputs['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(ids=ids, mask=mask).view(-1)
                #                 print(outputs.shape)
                #                 print(targets.shape)
                loss = self.loss_fn(outputs, targets)
                prog_bar.set_description('loss: {:.2f}'.format(loss.item()))

                Config.scaler.scale(loss).backward()
                Config.scaler.step(self.optimizer)
                Config.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

    def valid_one_epoch(self):
        """
        This function validates the model for one epoch through all batches of the valid dataset
        It also returns the validation Root mean squared error for assesing model performance.
        """
        prog_bar = tqdm(enumerate(self.valid_data), total=len(self.valid_data))
        self.model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for idx, inputs in prog_bar:
                ids = inputs['ids'].to(self.device, dtype=torch.long)
                mask = inputs['mask'].to(self.device, dtype=torch.long)

                targets = inputs['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(ids=ids, mask=mask)
                all_targets.extend(targets.cpu().detach().numpy().tolist())
                all_predictions.extend(outputs.cpu().detach().numpy().tolist())
        #                 print(all_targets.size)
        #                 print(all_predictions.size)
        val_rmse_loss = np.sqrt(mean_squared_error(all_targets, all_predictions))
        print('Validation RMSE: {:.2f}'.format(val_rmse_loss))

        return val_rmse_loss

    def get_model(self):
        return self.model

    def yield_optimizer(model):
        """
        Returns optimizer for specific parameters
        """
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return transformers.AdamW(optimizer_parameters, lr=Config.LR)


