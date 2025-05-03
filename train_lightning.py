import torch
from engine_lightning import LitEngineResNet
from lartpcdataset import lartpcDataset
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger

# Setting the maximum of training epochs
n_max_epochs = 10


wandb_logger = WandbLogger(project='tutorial-resnet-lightning')

DEVICE = torch.device("cuda")
#DEVICE = torch.device("cpu")

BATCHSIZE=64

# Defining the training and validation datasets
train_dataset = lartpcDataset( root="./data/z-view/" )
valid_dataset = lartpcDataset( root="./valid/z-view/" )

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCHSIZE,
    shuffle=True)
val_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=BATCHSIZE,
    shuffle=False)


# model
model = LitEngineResNet().to(DEVICE)
model.print_model()

# testing block
if False:
    print("//////// TESTING BLOCK //////////")
    imgs, labels = next( iter(train_loader) )
    imgs = imgs.to(DEVICE)
    labels = labels.to(DEVICE)
    print("imgs: ",imgs.shape)
    print("labels: ",labels.shape," :: ",labels)
    out = model.forward(imgs)
    print(out.shape)

    loss = model.calc_loss( out, labels )
    print(loss)

# training
trainer = pl.Trainer(accelerator="gpu",
                     devices=1,
                     precision=16,
                     max_epochs=n_max_epochs,
                     limit_train_batches=0.5,
                     logger=wandb_logger)
trainer.fit(model, train_loader, val_loader)


