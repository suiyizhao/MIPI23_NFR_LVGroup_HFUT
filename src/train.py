import time
import torch
import random

import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from utils import *
from options import TrainOptions
from models import NAFNet
from losses import LossCont, LossFreqReco
from datasets import Flare_Image_Loader, SingleImgDataset

print('---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------')
opt = TrainOptions().parse()

set_random_seed(opt.seed)

models_dir, log_dir, train_images_dir, val_images_dir = prepare_dir(opt.results_dir, opt.experiment, delete=(not opt.resume))

writer = SummaryWriter(log_dir=log_dir)

print('---------------------------------------- step 2/5 : data loading... ------------------------------------------------')
print('training data loading...')
train_dataset = Flare_Image_Loader(data_source=opt.data_source + '/train', crop=opt.crop)
train_dataset.load_scattering_flare()
train_dataloader = DataLoader(train_dataset, batch_size=opt.train_bs, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
print('successfully loading training pairs. =====> qty:{} bs:{}'.format(len(train_dataset),opt.train_bs))

print('validating data loading...')
val_dataset = SingleImgDataset(data_source=opt.data_source + '/val')
val_dataloader = DataLoader(val_dataset, batch_size=opt.val_bs, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
print('successfully loading validating pairs. =====> qty:{} bs:{}'.format(len(val_dataset),opt.val_bs))

print('---------------------------------------- step 3/5 : model defining... ----------------------------------------------')
model = NAFNet().cuda()

if opt.data_parallel:
    model = nn.DataParallel(model)
print_para_num(model)

if opt.pretrained is not None:
    model.load_state_dict(torch.load(opt.pretrained))
    print('successfully loading pretrained model.')
    
print('---------------------------------------- step 4/5 : requisites defining... -----------------------------------------')
criterion_cont = LossCont()
criterion_fft = LossFreqReco()

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50,100,150,200,250,300], 0.5)

print('---------------------------------------- step 5/5 : training... ----------------------------------------------------')
def main():
    
    start_epoch = 1
    if opt.resume:
        state = torch.load(models_dir + '/latest.pth')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch'] + 1
        print('Resume from epoch %d' % (start_epoch))
    
    for epoch in range(start_epoch, opt.n_epochs + 1):
        train(epoch)
        
        if (epoch) % opt.val_gap == 0:
            val(epoch)
        
    writer.close()
    
def train(epoch):
    model.train()
    
    max_iter = len(train_dataloader)

    psnr_meter = AverageMeter()    
    iter_cont_meter = AverageMeter()
    iter_fft_meter = AverageMeter()
    iter_timer = Timer()
    
    for i, (gts, flares, imgs, _) in enumerate(train_dataloader):
        gts, flares, imgs = gts.cuda(), flares.cuda(), imgs.cuda()
        cur_batch = imgs.shape[0]
        
        optimizer.zero_grad()
        preds_flare, preds = model(imgs)
        
        loss_cont = criterion_cont(preds, gts) + 0.1*criterion_cont(preds_flare, flares)
        loss_fft = criterion_fft(preds, gts) + 0.1*criterion_fft(preds_flare, flares)
        loss = loss_cont + opt.lambda_fft * loss_fft
        
        loss.backward()
        optimizer.step()
        
        psnr_meter.update(get_metrics(torch.clamp(preds.detach(), 0, 1), gts), cur_batch)
        iter_cont_meter.update(loss_cont.item()*cur_batch, cur_batch)
        iter_fft_meter.update(loss_fft.item()*cur_batch, cur_batch)
        
        if i == 0:
            save_image(torch.cat((imgs,preds.detach(),preds_flare.detach(),flares,gts),0), train_images_dir + '/epoch_{:0>4}_iter_{:0>4}.png'.format(epoch, i+1), nrow=opt.train_bs, normalize=True, scale_each=True)
            
        if (i+1) % opt.print_gap == 0:
            print('Training: Epoch[{:0>4}/{:0>4}] Iteration[{:0>4}/{:0>4}] Loss_cont: {:.4f} Loss_fft: {:.4f} PSNR: {:.4f} Time: {:.4f}'.format(epoch, opt.n_epochs, i + 1, max_iter, iter_cont_meter.average(), iter_fft_meter.average(), psnr_meter.average(), iter_timer.timeit()))
            writer.add_scalar('PSNR', psnr_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
            writer.add_scalar('Loss_cont', iter_cont_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
            writer.add_scalar('Loss_fft', iter_fft_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
            
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch}, models_dir + '/latest.pth')
    scheduler.step()
    
def val(epoch):
    model.eval()
    
    print(''); print('Validating...', end=' ')
    
    timer = Timer()
    
    for i, (img, path) in enumerate(val_dataloader):
        img = img.cuda()

        with torch.no_grad():
            pred_flare, pred = model(img)
        pred_clip = torch.clamp(pred, 0, 1)
        pred_flare_clip = torch.clamp(pred_flare, 0, 1)
        
        if i < 5:
            # save_image(pred_clip, val_images_dir + '/epoch_{:0>4}_'.format(epoch) + os.path.basename(path[0]))
            save_image(pred_clip, val_images_dir + '/epoch_{:0>4}_img_'.format(epoch) + os.path.basename(path[0]), nrow=opt.val_bs//2, normalize=True, scale_each=True)
            save_image(pred_flare_clip, val_images_dir + '/epoch_{:0>4}_flare_'.format(epoch) + os.path.basename(path[0]), nrow=opt.val_bs//2, normalize=True, scale_each=True)
        else:
            break

    torch.save(model.state_dict(), models_dir + '/epoch_{:0>4}.pth'.format(epoch))

    print('Epoch[{:0>4}/{:0>4}] Time: {:.4f}'.format(epoch, opt.n_epochs, timer.timeit())); print('')

    
if __name__ == '__main__':
    main()
    