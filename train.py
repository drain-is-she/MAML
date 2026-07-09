import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import *
from torchvision.datasets import ImageFolder
from transforms import train_transform, val_transform
from models.network import MammalClassifier
from losses import ClassificationLoss
from metrics import accuracy
from utils import (
    set_seed,
    AverageMeter,
    save_checkpoint,
    get_lr
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        _, logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        acc = accuracy(logits, labels)

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        _, logits = model(images)

        loss = criterion(logits, labels)

        acc = accuracy(logits, labels)

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

    return loss_meter.avg, acc_meter.avg


def main():

    set_seed(SEED)

    train_dataset = ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = ImageFolder(
        VAL_DIR,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = MammalClassifier(pretrained=True).to(DEVICE)

    # -----------------------------
    # Phase 1: Train classifier only
    # -----------------------------
    model.freeze_backbone()

    criterion = ClassificationLoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_MAX,
        eta_min=ETA_MIN
    )

    best_acc = 0.0

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {get_lr(optimizer):.6f}"
        )

        if val_acc > best_acc:

            best_acc = val_acc

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                path=os.path.join(
                    CHECKPOINT_DIR,
                    CHECKPOINT_NAME
                )
            )

            print("Best model saved.")

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
