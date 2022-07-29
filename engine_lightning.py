import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import resnet
import pytorch_lightning as pl
import wandb

class LitEngineResNet(pl.LightningModule):
    def __init__(self,pretrained=False):
        super().__init__()
        #self.wandb
        input_channels = 1
        if pretrained:
            # works on RGB images
            input_channels = 3
        self.model = resnet.resnet18( pretrained=pretrained,
                                      input_channels=input_channels,
                                      num_classes=5)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def print_model(self):
        print(self.model)
        
    def forward(self,x):
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch # data batch, labels
        z = self.model(x) 

        loss = self.calc_loss( z, y )
        self.log('train_loss', loss)
        return loss

    def calc_loss( self, pred, labels ):
        loss = self.loss_fn( pred, labels )
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.model(x)
        loss = self.calc_loss( z, y )
        self.log('val_loss', loss)
