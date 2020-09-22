import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np

import torch
from torch.autograd import Variable
import tqdm


def variable(x, volatile=False):
    with torch.no_grad():
        if isinstance(x, (list, tuple)):
            return [variable(y, volatile=volatile) for y in x]
        return cuda(x)


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)

    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    best_model_path = root / 'best_model_{fold}.pt'.format(fold=fold)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']

        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    if best_model_path.exists():
        best_state = torch.load(str(best_model_path))
        best_val_loss = best_state['val_loss']
    else:
        best_val_loss = np.inf

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    save_best = lambda ep, val_loss: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'val_loss': val_loss,
    }, str(best_model_path))

    report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if valid_loss < best_val_loss:
                save_best(epoch + 1, valid_loss)
                best_val_loss = valid_loss
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return

def train_multi(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)

    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    best_model_path = root / 'best_model_{fold}.pt'.format(fold=fold)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']

        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    if best_model_path.exists():
        best_state = torch.load(str(best_model_path))
        best_val_loss = best_state['val_loss']
    else:
        best_val_loss = np.inf

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step
    }, str(model_path))

    save_best = lambda ep, val_loss: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'val_loss': val_loss,
    }, str(best_model_path))

    report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    sloss_log = root.joinpath('train_sloss_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    closs_log = root.joinpath('train_closs_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        slosses = []
        closses = []
        tl = train_loader
        try:
            mean_loss = 0
            smean_loss = 0
            cmean_loss = 0
            for i, (inputs, targets, ctargets) in enumerate(tl):
                inputs, targets, ctargets = variable(inputs), variable(targets), variable(ctargets)
                outputs, coutputs = model(inputs)
                sloss, closs = criterion(outputs, targets, coutputs, ctargets)
                loss = sloss + closs
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                slosses.append(sloss.item())
                closses.append(closs.item())
                mean_loss = np.mean(losses[-report_each:])
                smean_loss = np.mean(slosses[-report_each:])
                cmean_loss = np.mean(closses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
                    write_event(sloss_log, step, loss=smean_loss)
                    write_event(closs_log, step, loss=cmean_loss)
            write_event(log, step, loss=mean_loss)
            write_event(sloss_log, step, loss=smean_loss)
            write_event(closs_log, step, loss=cmean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, step, **valid_metrics)
            write_event(sloss_log, step, **valid_metrics)
            write_event(closs_log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if valid_loss < best_val_loss:
                save_best(epoch + 1, valid_loss)
                best_val_loss = valid_loss
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
