import random
import glob

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import numpy as np
from PIL import Image

import plotly.express as px
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchmetrics
import pytorch_lightning as pl
import timm
from tqdm import tqdm

import autoaugment

random_seed = 42

from typing import Optional, List

class Unet(nn.Module):
    """Unet is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        num_classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        center: if ``True`` add ``Conv2dReLU`` block on encoder head
    NOTE: This is based off an old version of Unet in https://github.com/qubvel/segmentation_models.pytorch
    """

    def __init__(
            self,
            backbone='resnet50',
            backbone_kwargs=None,
            backbone_indices=None,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_chans=1,
            num_classes=5,
            center=False,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        # NOTE some models need different backbone indices specified based on the alignment of features
        # and some models won't have a full enough range of feature strides to work properly.
        encoder = timm.create_model(
            backbone, features_only=True, out_indices=backbone_indices, in_chans=in_chans,
            pretrained=True, **backbone_kwargs)
        encoder_channels = encoder.feature_info.channels()[::-1]
        self.encoder = encoder

        if not decoder_use_batchnorm:
            norm_layer = None
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=num_classes,
            norm_layer=norm_layer,
            center=center,
        )

    def forward(self, x: torch.Tensor):
        l = self.encoder(x)
        l.reverse()  # torchscript doesn't work with [::-1]
        x = self.decoder(l)
        return l[0].flatten(start_dim=1), x


class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_channels)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2.0, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, act_layer=act_layer)
        self.scale_factor = scale_factor
        if norm_layer is None:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels,  **conv_args)
        else:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, norm_layer=norm_layer, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, norm_layer=norm_layer, **conv_args)

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        if self.scale_factor != 1.0:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            norm_layer=nn.BatchNorm2d,
            center=False,
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = DecoderBlock(channels, channels, scale_factor=1.0, norm_layer=norm_layer)
        else:
            self.center = nn.Identity()

        in_channels = [in_chs + skip_chs for in_chs, skip_chs in zip(
            [encoder_channels[0]] + list(decoder_channels[:-1]),
            list(encoder_channels[1:]) + [0])]
        out_channels = decoder_channels

        self.blocks = nn.ModuleList()
        for in_chs, out_chs in zip(in_channels, out_channels):
            self.blocks.append(DecoderBlock(in_chs, out_chs, norm_layer=norm_layer))
        self.final_conv = nn.Conv2d(out_channels[-1], final_channels, kernel_size=(1, 1))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: List[torch.Tensor]):
        encoder_head = x[0]
        skips = x[1:]
        x = self.center(encoder_head)
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        x = self.final_conv(x)
        return x


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, noise, transforms):
        self.inputs = inputs
        self.noise = noise
        self.transforms = transforms

    def mix_up(self, input, noise, alpha=0.9):
        mixed_img = alpha * input + (1 - alpha) * noise

        return mixed_img

    def __getitem__(self, idx):
        anchor = self.transforms(Image.open(self.inputs[idx]))

        p_idx = random.randint(0, len(self.noise) - 1)
        noise = self.transforms(Image.open(self.noise[p_idx]))
        positive = self.mix_up(anchor, noise)

        n_idx = random.randint(0, len(self.noise) - 1)
        while n_idx == p_idx:
            n_idx = random.randint(0, len(self.noise) - 1)
        negative = self.transforms(Image.open(self.noise[n_idx]))

        return anchor, positive, negative

    def __len__(self):
        return len(self.inputs)


class Datamodule(pl.LightningDataModule):
    def __init__(self, data_name, batch_size):
        super().__init__()

        self.data_name = data_name
        self.batch_size = batch_size

    def setup(self, stage):
        if self.data_name == 'cifar':
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                # autoaugment.SVHNPolicy(fillcolor=(128)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5), (0.5)),
            ])

            train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=train_transform, download=True)
            _, train_idx = train_test_split(
                np.arange(len(train_dataset)),
                test_size=100,
                random_state=random_seed,
                shuffle=True,
                stratify=train_dataset.targets
            )
            self.train_sub_dataset = torch.utils.data.Subset(train_dataset, train_idx)

            test_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5), (0.5)),
            ])
            self.test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transform, download=True)
            _, test_idx = train_test_split(
                np.arange(len(self.test_dataset)),
                test_size=1000,
                random_state=random_seed,
                shuffle=True,
                stratify=self.test_dataset.targets
            )
            self.test_sub_dataset = torch.utils.data.Subset(self.test_dataset, test_idx)
        elif self.data_name == 'mnist':
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                # autoaugment.SVHNPolicy(fillcolor=(128)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5), (0.5)),
                # torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])

            train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
            _, train_idx = train_test_split(
                np.arange(len(train_dataset)),
                test_size=10,
                random_state=random_seed,
                shuffle=True,
                stratify=train_dataset.targets.tolist()
            )
            self.train_sub_dataset = torch.utils.data.Subset(train_dataset, train_idx)

            test_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5), (0.5)),
                # torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
            self.test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=test_transform, download=True)
            _, test_idx = train_test_split(
                np.arange(len(self.test_dataset)),
                test_size=100,
                random_state=random_seed,
                shuffle=True,
                stratify=self.test_dataset.targets.tolist()
            )
            self.test_sub_dataset = torch.utils.data.Subset(self.test_dataset, test_idx)
        else:
            test_inputs = glob.glob('data/nc_001/test/*.jpg')
            train_inputs = glob.glob('data/nc_001/train/*.jpg')
            random.shuffle(train_inputs)
            val_inputs = train_inputs[:len(test_inputs)]
            train_inputs = train_inputs[len(test_inputs):]

            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                # autoaugment.SVHNPolicy(fillcolor=(128)),
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize((0.5), (0.5)),
                # torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])

            self.train_sub_dataset = Dataset(train_inputs, train_inputs, train_transform)

            test_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize((0.5), (0.5)),
                # torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
            self.test_dataset = Dataset(train_inputs, test_inputs, test_transform)
            self.test_sub_dataset = Dataset(train_inputs, val_inputs, test_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_sub_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)

    def train_dataloader_no_shuffle(self):
        return torch.utils.data.DataLoader(self.train_sub_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_sub_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=int(self.batch_size), shuffle=False, num_workers=16)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=int(self.batch_size), shuffle=False, num_workers=16)


class Model(pl.LightningModule):
    def __init__(self, model_name, data_name):
        super().__init__()
        self.model_name = model_name
        self.data_name = data_name
        if data_name == 'cifar':
            in_chans = 3
        else:
            in_chans = 1

        # self.model = timm.create_model(model_name, pretrained=False, num_classes = 1000)
        self.model = Unet(backbone=model_name, in_chans=in_chans, num_classes=in_chans)
        # self.classifier = torch.nn.Linear(in_chans*224*224, 100)

        # self.cel = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()
        self.tml = torch.nn.TripletMarginLoss(margin=10)
        self.dist = torch.nn.PairwiseDistance(p=2)
        # self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=100)

    def training_step(self, batch, batch_idx):
        a, p, n = batch

        al, ai = self.model(a)
        pl, pi = self.model(p)
        nl, ni = self.model(n)

        a_loss = self.mse(ai, a)
        p_loss = self.mse(pi, p)
        n_loss = self.mse(ni, n)
        t_loss = self.tml(al, pl, nl)

        img_loss = a_loss + p_loss + n_loss
        loss = img_loss + t_loss

        self.log("img_loss", img_loss, prog_bar=True)
        self.log("tml_loss", t_loss, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        a, p, n = batch

        al, ai = self.model(a)
        pl, pi = self.model(p)
        nl, ni = self.model(n)

        t_loss = self.tml(al, pl, nl)

        self.log("val_loss", t_loss, prog_bar=True)

        rand_idx = random.randint(0, a.size(0) - 1)
        img = Image.fromarray((255 * ai[rand_idx][0]).detach().cpu().numpy().astype(np.uint8))
        org = Image.fromarray((255 * a[rand_idx][0]).detach().cpu().numpy().astype(np.uint8))
        self.logger.log_image("G", [img, org], caption=['G', 'O'])

    def test_step(self, batch, batch_idx):
        a, p, n = batch

        al, ai = self.model(a)
        pl, pi = self.model(p)
        nl, ni = self.model(n)

        p_dist = self.dist(al, pl)
        n_dist = self.dist(al, nl)

        self.log("test_ap_dist", p_dist.mean())
        self.log("test_an_dist", n_dist.mean())

    def evaluate(self, test_dataloader):
        p_dists = []
        n_dists = []
        for a, p, n in tqdm(test_dataloader, desc='get test data', total=len(test_dataloader)):
            al, ai = self.model(a)
            pl, pi = self.model(p)
            nl, ni = self.model(n)

            p_dist = self.dist(al, pl)
            n_dist = self.dist(al, nl)

            p_dists.append(p_dist.detach().cpu())
            n_dists.append(n_dist.detach().cpu())

        p_dists = torch.cat(p_dists).mean()
        n_dists = torch.cat(n_dists).mean()

        print(f"p_dists : {p_dists.item()}")
        print(f"n_dists : {n_dists.item()}")

    def visualize(self, test_dataloader):
        self.model = self.model.cuda()
        vectors = []
        labels = []
        colors = []

        label_count = 0
        for a, p, n in tqdm(test_dataloader, desc='get test data', total=len(test_dataloader)):
            al, ai = self.model(a.cuda())
            vectors.append(al.detach().cpu())
            labels += [label_count + i for i in range(al.size(0))]
            colors += [0] * al.size(0)

            pl, pi = self.model(p.cuda())
            vectors.append(pl.detach().cpu())
            labels += [label_count + i for i in range(pl.size(0))]
            colors += [1] * pl.size(0)

            label_count += al.size(0)

        vectors = torch.cat(vectors)

        tsne = TSNE(n_components=2, random_state=42)
        embedded_data = tsne.fit_transform(vectors)

        fig = px.scatter(x=embedded_data[:, 0], y=embedded_data[:, 1],
                         color=colors, labels={'color': 'Label'},
                         title='2D Scatter Plot with Hover Labels',
                         hover_name=labels,
                         text=labels)  # 호버 시 표시될 텍스트 (라벨)
        fig.show()
        fig.write_html("vis.html")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer

data_name = 'nc001'
model_type = 'UT-R50'
model_name = 'resnet50'
batch_size = 32
max_epochs = 1

datamodule = Datamodule(data_name, batch_size)
datamodule.setup('fit')
model = Model(model_name, data_name)

logger = pl.loggers.WandbLogger(project='ksc_brand', name=f"{model_type}_mixup1")

trainer = pl.Trainer(accelerator='gpu', devices=[7], max_epochs=max_epochs, callbacks=[pl.callbacks.ModelCheckpoint(dirpath='model', monitor='val_loss', filename=model_type+'_mixup-{epoch:02d}-{val_loss:.2f}', save_top_k=1), pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=30)], logger=logger, log_every_n_steps=1)
trainer.fit(model=model, datamodule=datamodule)
# trainer.test(model=model, datamodule=datamodule)
# model.evaluate(datamodule.test_dataloader())
model.visualize(datamodule.test_dataloader())

wandb.finish()