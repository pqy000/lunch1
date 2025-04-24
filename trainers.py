# trainers.py

import torch
import random
import torchvision.transforms as T

def train_ER(model, criterion, optimizer, train_loader, replay_mem, device, replay_batch=32):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        logits = model(imgs)
        loss_main = criterion(logits, labels)

        re_data = replay_mem.get_batch(batch_size=replay_batch)
        if re_data is not None:
            rx, ry, _ = re_data
            rx, ry = rx.to(device), ry.to(device)
            logits_re = model(rx)
            loss_re = criterion(logits_re, ry)
        else:
            loss_re = torch.tensor(0.0, device=device)

        loss = loss_main + loss_re
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        replay_mem.add_samples(imgs, labels)


def train_DER(model, criterion, optimizer, train_loader, replay_mem, device,
              replay_batch=32, alpha=0.1):
    model.train()
    mse = torch.nn.MSELoss()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        logits_new = model(imgs)
        loss_main = criterion(logits_new, labels)

        re_data = replay_mem.get_batch(batch_size=replay_batch)
        if re_data is not None:
            rx, ry, rlogits = re_data
            rx, ry = rx.to(device), ry.to(device)
            logits_cur = model(rx)
            loss_old_ce = criterion(logits_cur, ry)
            loss_mse = torch.tensor(0.0, device=device)
            if rlogits is not None:
                loss_mse = mse(logits_cur, rlogits.to(device))
        else:
            loss_old_ce = torch.tensor(0.0, device=device)
            loss_mse = torch.tensor(0.0, device=device)

        loss = loss_main + loss_old_ce + alpha * loss_mse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store logits
        with torch.no_grad():
            stored_logits = logits_new.detach().cpu()
        replay_mem.add_samples(imgs, labels, logits=stored_logits)


def train_RAR(model, criterion, optimizer, train_loader, replay_mem, device,
              replay_batch=32, repeat_times=2):
    model.train()
    augmentation = T.RandomHorizontalFlip(p=0.5)
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        logits = model(imgs)
        loss_main = criterion(logits, labels)

        total_loss_re = 0.0
        re_data = replay_mem.get_batch(batch_size=replay_batch)
        if re_data is not None:
            rx, ry, _ = re_data
            for _ in range(repeat_times):
                rx_aug = []
                for i in range(rx.size(0)):
                    xpil = T.ToPILImage()(rx[i])
                    xaug = augmentation(xpil)
                    rx_aug.append(T.ToTensor()(xaug))
                rx_aug = torch.stack(rx_aug).to(device)
                ry = ry.to(device)
                logits_re = model(rx_aug)
                loss_re = criterion(logits_re, ry)
                total_loss_re += loss_re
            total_loss_re /= repeat_times
        else:
            total_loss_re = torch.tensor(0.0, device=device)

        loss = loss_main + total_loss_re
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        replay_mem.add_samples(imgs, labels)