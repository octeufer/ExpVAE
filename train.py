import torch
import torchvision
import time, os, datetime
import copy
from helper import group_eval

from load import log
from torch.distributions.bernoulli import Bernoulli

import matplotlib.pyplot as plt
import torch.nn.functional as F

# from operators import *

def train_model(model, dataloaders, optimizer, scheduler, num_epochs, device, outdir, cfg, logfile):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    PATH = outdir
    global_epoch = num_epochs

    for epoch in range(num_epochs):
        log(logfile, ("Epoch: %f/%f" % (epoch, num_epochs - 1)))
        log(logfile, ("----------"))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
        
            running_loss = 0.0
            running_corrects = 0

            for ibatch, batch in enumerate(dataloaders[phase]):
                length = len(batch)
                if length == 2:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                elif length == 3:
                    inputs, labels, g = batch
                    inputs, labels, g = inputs.to(device), labels.to(device), g.to(device)
                else:
                    inputs, labels, g, c, sp = batch
                    inputs, labels, c = inputs.to(device), labels.to(device), c.to(device)
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        # print(inputs.size())
                        mu, logvar = model.encode(inputs, labels, c)
                        z = model.reparameterize(mu, logvar)
                        y_recon = model.decode_y(z)
                        c_recon = model.decode_c(z)
                        x_recon = model.decode_x(z, y_recon)
                        y_pred = model.predict(x_recon)
                        loss = model.loss_function(x_recon, c_recon, y_recon, inputs, labels, c, y_pred, mu, logvar, cfg)
                        # print(loss)
                        loss_total = loss['total']
                        # outputs= model(inputs)
                        # loss = criterion(outputs, labels)
                        # _, preds = torch.max(outputs, 1)
                        _, preds = torch.max(y_pred, 1)
                        objective = loss_total
                        objective.backward()
                        optimizer.step()

                    if phase == 'val':
                        mu, logvar = model.encode(inputs, labels, c)
                        z = model.reparameterize(mu, logvar)
                        y_recon = model.decode_y(z)
                        c_recon = model.decode_c(z)
                        x_recon = model.decode_x(z, y_recon)
                        y_pred = model.predict(x_recon)
                        loss = model.loss_function(x_recon, c_recon, y_recon, inputs, labels, c, y_pred, mu, logvar, cfg)
                        # print(loss)
                        loss_total = loss['total']
                        # outputs= model(inputs)
                        # loss = criterion(outputs, labels)
                        _, preds = torch.max(y_pred, 1)
                        # preds = y_recon.sample()
                
                running_loss += loss['pred_y'].item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if ibatch % 1000 == 0:
                    log(logfile, ("current: %s" % (ibatch * len(inputs))))
                    log(logfile, ("reconstruction X loss: %s" % (loss['recon_x'])))
                    log(logfile, ("kld loss: %s" % (loss['kld'])))
                    log(logfile, ("pred loss: %s" % (loss['pred_y'])))
                    log(logfile, ("context loss: %s" % (loss['recon_c'])))
                    log(logfile, ("class loss: %s" % (loss['recon_y'])))
                    log(logfile, ("======================================"))
                    torchvision.utils.save_image(inputs.data, os.path.join(logfile[:-7], 'vis_ori_{}.jpg'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f"))), nrow=8, padding=2)
                    torchvision.utils.save_image(x_recon.data, os.path.join(logfile[:-7], 'vis_recon_{}.jpg'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f"))), nrow=8, padding=2)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == 'val':
                scheduler.step(epoch_loss)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            log(logfile, ("Epoch: %f/%f" % (epoch, num_epochs - 1)))
            log(logfile, ('%s Pred Loss: %f Pred Acc: %f') % (phase, epoch_loss, epoch_acc))
            log(logfile, ('%s Reconstruction X Loss: %f') % (phase, loss['recon_x']))
            log(logfile, ('%s KLD Loss: %f Total Loss: %f') % (phase, loss['kld'], loss['total']))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        global_epoch -= 1

        if epoch % 10 == 0:
            PATH = outdir + 'model_' + str(epoch)
            torch.save({
            'epoch': global_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best val acc': best_acc,
            }, PATH)

        log(logfile, ('----------'))
 
 
    time_elapsed = time.time() - since
    log(logfile, ("Training complete in %sm %ss" % (time_elapsed // 60, time_elapsed % 60)))
    log(logfile, ("Best val Acc: %s" % best_acc))

    model.load_state_dict(best_model_wts)
    PATH = outdir + 'model_' + str(global_epoch)
    torch.save({
            'epoch': global_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best val acc': best_acc,
            }, PATH)
    return model, val_acc_history


def test_model(dataloader, trainmodel, device, config, logfile):
    size = len(dataloader.dataset)
    trainmodel.eval()
    test_loss, correct = 0, 0
    test_loss_group, correct_group = {}, {}
    group_size = {}

    # with torch.no_grad():
    for ibatch, batch in enumerate(dataloader):
        length = len(batch)
        if length == 2:
            X, y = batch
            X, y = X.to(device), y.to(device)
        elif length == 3:
            X, y, g = batch
            # inputs = group_eval(X, y, g)
            # # print(inputs.keys())
            # for key in inputs.keys():
            #     if key not in test_loss_group:
            #         test_loss_group[key] = 0
            #         correct_group[key] = 0
            #         group_size[key] = 0
            #     X_temp, y_temp = inputs[key]
            #     X_temp = torch.stack(X_temp)
            #     y_temp = torch.stack(y_temp)
            #     X_temp, y_temp = X_temp.to(device), y_temp.to(device)
            #     group_size[key] += X_temp.size(0)
            #     outputs_temp = trainmodel(X_temp)
            #     loss_temp = criterion(outputs_temp, y_temp)
            #     _, preds_temp = torch.max(outputs_temp, 1)

            #     test_loss_group[key] += loss_temp.item() * X_temp.size(0)
            #     correct_group[key] += torch.sum(preds_temp == y_temp.data)
            X, y, g = X.to(device), y.to(device), g.to(device)
        else:
            X, y, g, c, sp = batch
            # inputs = group_eval(X, y, g)
            # print(inputs.keys())
            # for key in inputs.keys():
            #     if key not in test_loss_group:
            #         test_loss_group[key] = 0
            #         correct_group[key] = 0
            #         group_size[key] = 0
            #     X_temp, y_temp, c_temp = inputs[key]
            #     X_temp = torch.stack(X_temp)
            #     y_temp = torch.stack(y_temp)
            #     c_temp = torch.stack(c_temp)
            #     X_temp, y_temp, c_temp = X_temp.to(device), y_temp.to(device), c_temp.to(device)
            #     group_size[key] += X_temp.size(0)
                
            #     mu_temp, logvar_temp = trainmodel.encode(X_temp, y_temp, c_temp)
            #     z_temp = trainmodel.reparameterize(mu_temp, logvar_temp)
            #     y_recon_temp = trainmodel.decode_y(z_temp)
            #     c_recon_temp = trainmodel.decode_c(z_temp)
            #     x_recon_temp = trainmodel.decode_x(z_temp, y_recon_temp)
            #     y_pred_temp = trainmodel.predict(x_recon)
            #     loss_temp = trainmodel.loss_function(x_recon_temp, c_recon_temp, y_recon_temp, inputs, y, c, y_pred, mu, logvar, cfg)
            #     test_loss_group[key] += loss_temp.item() * X_temp.size(0)
            #     correct_group[key] += torch.sum(preds_temp == y_temp.data)
            X, y, g, c = X.to(device), y.to(device), g.to(device), c.to(device)

        mu, logvar = trainmodel.encode(X, y, c)
        z = trainmodel.reparameterize(mu, logvar)
        y_recon = trainmodel.decode_y(z)
        c_recon = trainmodel.decode_c(z)
        x_recon = trainmodel.decode_x(z, y_recon)
        y_pred = trainmodel.predict(x_recon)
        loss = trainmodel.loss_function(x_recon, c_recon, y_recon, X, y, c, y_pred, mu, logvar, config)
        # print(loss)
        loss_total = loss['total']
        # outputs= model(inputs)
        # loss = criterion(outputs, labels)
        _, preds = torch.max(y_pred, 1)

        test_loss += loss['pred_y'].item() * X.size(0)
        correct += torch.sum(preds == y.data)

        # with torch.no_grad():

        if ibatch % 100 == 0:
            log(logfile, ("current: %s" % (ibatch * len(X))))

        sampler = Bernoulli(probs=0.01)
        flag = sampler.sample()
        if flag:
            # imgs = x_recon.permute(0,2,3,1).cpu().data.numpy()
            # print(imgs[0])
            # fig, ax = plt.subplots(dpi=100, frameon=False)
            # fig.suptitle(y[0])
            # ax.imshow(imgs[0].squeeze())
            # ax.axis('off')
            # filename_gen = 'vis_gen{}.png'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f"), 0)
            # plt.savefig(os.path.join(logfile[:-7], filename_gen), dpi=100)
            # imgs_ori = X.permute(0,2,3,1).cpu().data.numpy()
            # print(imgs_ori[0])
            # fig, ax = plt.subplots(dpi=100, frameon=False)
            # fig.suptitle(y[0])
            # ax.imshow(imgs_ori[0].squeeze())
            # ax.axis('off')    
            # filename_ori = 'vis_ori{}.png'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f"), 0)
            # plt.savefig(os.path.join(logfile[:-7], filename_ori), dpi=100)
            # plt.close()
            rec = torch.nn.MSELoss()
            rec_x = rec(x_recon, X)
            # print('min X: %f, max X: %f' % (torch.min(X), torch.max(X)))
            print('rec: %f' % (rec_x.data))
            y_p = trainmodel.predict(x_recon)
            _, ps = torch.max(y_p, 1)
            corr = torch.sum(ps == y.data)
            print('corr: %d' % (corr.data))
            torchvision.utils.save_image(X.data, os.path.join(logfile[:-7], 'vis_ori_{}.jpg'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f"))), nrow=8, padding=2)
            torchvision.utils.save_image(x_recon.data, os.path.join(logfile[:-7], 'vis_recon_{}.jpg'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f"))), nrow=8, padding=2)


    correct = correct.double()
    test_loss /= size
    correct /= size
    log(logfile, (f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"))
    
    for key in test_loss_group.keys():
        test_loss_g = test_loss_group[key]
        correct_g = correct_group[key].double()
        test_loss_g /= group_size[key]
        correct_g /= group_size[key]
        log(logfile, (f"Test Error Group {key}: \n Accuracy: {(100*correct_g):>0.1f}%, Avg loss: {test_loss_g:>8f} \n"))


def test_model_counterfactual(dataloader, dataset, trainmodel, device, criterion, config, logfile):
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    Tensor = FloatTensor

    tv_beta = 3
    learning_rate = 0.1
    max_iterations = 500
    l1_coeff = 0.01
    tv_coeff = 0.2

    size = len(dataloader.dataset)
    trainmodel.eval()
    test_loss, correct = 0, 0
    test_loss_group, correct_group = {}, {}
    group_size = {}

    for ibatch, batch in enumerate(dataset):
        X, y, g = batch
        X, y, g = X.to(device), y.to(device), g.to(device)

        outputs = trainmodel(X)
        loss = criterion(outputs, y)
        _, preds = torch.max(outputs, 1)

        test_loss += loss.item() * X.size(0)
        correct += torch.sum(preds == y.data)