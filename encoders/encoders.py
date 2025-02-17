import torch.optim as optim
import pytorch_lightning as pl
import timm
from torchmetrics import Accuracy, Precision, Recall, F1Score
import torch



class timm_backbones(pl.LightningModule):
    def __init__(self, encoder='resnet18', num_classes=2, optimizer_cfg=None, l1_lambda=0.0):
        super().__init__()

        self.encoder = encoder
        self.model = timm.create_model(encoder, pretrained=True)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes)
        self.recall = Recall(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes)

        self.l1_lambda = l1_lambda
        if hasattr(self.model, 'fc'):  # For models with 'fc' as the classification layer
            in_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'classifier'):  # For models with 'classifier'
            in_features = self.model.classifier.in_features
            self.model.classifier = torch.nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'head'):  # For models with 'head'
            in_features = self.model.head.in_features
            self.model.head = torch.nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model architecture for encoder: {encoder}")

        if optimizer_cfg is not None:
            optimizer_name = optimizer_cfg.name
            optimizer_lr = optimizer_cfg.lr
            optimizer_weight_decay = optimizer_cfg.weight_decay

            if optimizer_name == 'Adam':
                self.optimizer = optim.Adam(self.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)
            elif optimizer_name == 'SGD':
                self.optimizer = optim.SGD(self.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10, min_lr=10e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):

        pass

    def on_validation_epoch_end(self):

        pass

    def test_step(self, batch, batch_idx):

        pass
    