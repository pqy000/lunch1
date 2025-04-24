# trainers_lunch.py

import torch
import random
import torchvision.transforms as T

from lunch_utils import KendallUncertaintyWeighting

def train_ER_LUNCH(model, lw_model, optimizer, train_loader, replay_mem, device,
                   replay_batch=32, ema=None):
    """
    ER + LUNCH
    - We consider two losses: L_new, L_old
    - lw_model: KendallUncertaintyWeighting with keys=["new","old"]
    - ema: ExponentialMovingAverage or None
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits_new = model(imgs)
        L_new = criterion(logits_new, labels)

        L_old = torch.tensor(0.0, device=device)
        re_data = replay_mem.get_batch(batch_size=replay_batch)
        if re_data is not None:
            rx, ry, _ = re_data
            rx, ry = rx.to(device), ry.to(device)
            logits_re = model(rx)
            L_old = criterion(logits_re, ry)

        # LUNCH combine
        loss_dict = {
            "new": L_new,
            "old": L_old
        }
        total_loss = lw_model(loss_dict)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # update EMA
        if ema is not None:
            ema.update(model)

        # add samples
        replay_mem.add_samples(imgs, labels)


def train_DER_LUNCH(model, lw_model, optimizer, train_loader, replay_mem, device,
                    replay_batch=32, alpha=0.1, ema=None):
    """
    DER + LUNCH
    We have 3 losses: L_new, L_oldCE, L_mse
    """
    criterion = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    model.train()

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits_new = model(imgs)
        L_new = criterion(logits_new, labels)

        L_oldCE = torch.tensor(0.0, device=device)
        L_mse   = torch.tensor(0.0, device=device)
        re_data = replay_mem.get_batch(batch_size=replay_batch)
        if re_data is not None:
            rx, ry, rlogits = re_data
            rx, ry = rx.to(device), ry.to(device)
            logits_cur = model(rx)
            L_oldCE = criterion(logits_cur, ry)
            if rlogits is not None:
                L_mse = mse(logits_cur, rlogits.to(device))

        # LUNCH combine
        loss_dict = {
            "new": L_new,
            "oldCE": L_oldCE,
            "mse": alpha*L_mse
        }
        total_loss = lw_model(loss_dict)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # update EMA
        if ema is not None:
            ema.update(model)

        # store logits
        with torch.no_grad():
            stored_logits = logits_new.detach().cpu()
        replay_mem.add_samples(imgs, labels, logits=stored_logits)


def train_RAR_LUNCH(model, lw_model, optimizer, train_loader, replay_mem, device,
                    replay_batch=32, repeat_times=2, ema=None):
    """
    RAR + LUNCH:
    2 losses: L_new, L_replay(aug)
    """
    criterion = torch.nn.CrossEntropyLoss()
    augmentation = T.RandomHorizontalFlip(p=0.5)
    model.train()

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        L_new = criterion(logits, labels)

        L_re = torch.tensor(0.0, device=device)
        re_data = replay_mem.get_batch(batch_size=replay_batch)
        if re_data is not None:
            rx, ry, _ = re_data
            # repeated augmentation
            for _ in range(repeat_times):
                rx_aug=[]
                for i in range(rx.size(0)):
                    xpil = T.ToPILImage()(rx[i])
                    xaug = augmentation(xpil)
                    rx_aug.append(T.ToTensor()(xaug))
                rx_aug = torch.stack(rx_aug).to(device)
                ry = ry.to(device)
                logits_re = model(rx_aug)
                L_re += criterion(logits_re, ry)
            L_re /= repeat_times

        loss_dict = {
            "new": L_new,
            "old": L_re
        }
        total_loss = lw_model(loss_dict)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # update EMA
        if ema is not None:
            ema.update(model)

        replay_mem.add_samples(imgs, labels)