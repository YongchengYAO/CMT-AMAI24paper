import torch
import numpy as np
from datasets import OAIZIB_Vt
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
import datetime
import os
import sys
from atlas_models import SVF_resid
import SimpleITK as sitk
from atlas_utils import *
import nibabel as nib

sys.path.append(os.path.realpath(".."))
import warnings
import argparse
import random
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='Aladdin Training')

# task name
parser.add_argument('--taskName', default="task1", type=str, help='task folder')
parser.add_argument('--jobName', default="run1", type=str, help='job name')

# folder
parser.add_argument('--dataFolder', type=str, help='data folder')
parser.add_argument('--modelFolder', type=str, help='model folder')
parser.add_argument('--TBFolder', type=str, help='tensorboard folder')

# continue training
parser.add_argument('--cont', action='store_true', default=False, help="continue training")
parser.add_argument('--logFile', type=str, help="path of the log files")
parser.add_argument('--templateImg', type=str, help="path of the saved template image")
parser.add_argument('--templateProb', type=str, help="path of the saved template probability map")

# model
parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', 
                    help='mini-batch size (default: 2), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--reg-factor', default=20000.0, type=float, help='regularization factor')
parser.add_argument('--sim-factor', default=10.0, type=float, help='similarity factor')
parser.add_argument('--template-pair-sim-factor', default=2.0, type=float,
                    help='pairwise similarity factor in template space')
parser.add_argument('--image-pair-sim-factor', default=5.0, type=float,
                    help='pairwise similarity factor in image space')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--sim-loss', default='SSD', type=str, help='Similarity Loss to use')
parser.add_argument('--save-per-epoch', default=5, type=int, help='number of epochs to save model')

# model training
# using GPU
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
parser.add_argument('--mps', action='store_true', default=False, help='Apple silicon acceleration')

if __name__ == "__main__":

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # [1] Model Configurations
    # =========================================================
    # device
    if args.gpu is not None:
        device = torch.device('cuda', args.gpu)
    elif args.mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # continue training
    if not args.cont:
        best_avgDice = 0.0
        epoch_start = 0
        max_epoch = args.epochs
    else:
        savedModelPath = os.path.join(args.modelFolder, 'checkpoints', 'model_last.pth.tar')
        if args.gpu is None:  # assuming the model was trained on GPU and debugged on CPU
            checkpoint = torch.load(savedModelPath, map_location=device)
        else:
            checkpoint = torch.load(savedModelPath)
        best_avgDice = checkpoint['best_avgDice']
        epoch_start = checkpoint['epoch']
        max_epoch = args.epochs
        if max_epoch <= epoch_start:
            raise ValueError('the maximum epoch shoulb be larger than that of the trained model')

    # other parameters
    batch_size = args.batch_size
    lr = args.lr
    loss_name = args.sim_loss
    reg_factor = args.reg_factor
    sim_factor = args.sim_factor
    template_pair_sim_factor = args.template_pair_sim_factor
    image_pair_sim_factor = args.image_pair_sim_factor
    # =========================================================

    # [2] Data
    # =========================================================
    # get subject list for each dataset
    dataFolder = args.dataFolder
    listPath_tr = os.path.join(dataFolder, 'train.txt')
    listPath_val = os.path.join(dataFolder, 'val.txt')
    listPath_ts = os.path.join(dataFolder, 'test.txt')
    train_list, valid_list, _ = getSubID_TrValTs(listPath_tr, listPath_val, listPath_ts)

    # construct data loader
    data_tr = OAIZIB_Vt(dataFolder, train_list)
    dataloader_tr = DataLoader(data_tr, batch_size=batch_size, shuffle=True, num_workers=4,
                               drop_last=True, pin_memory=True)
    data_template_update = OAIZIB_Vt(dataFolder, train_list)
    dataloader_template_update = DataLoader(data_template_update, batch_size=1, shuffle=False, num_workers=4,
                                            drop_last=True, pin_memory=True)
    data_val = OAIZIB_Vt(dataFolder, valid_list)
    dataloader_val = DataLoader(data_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    # =========================================================

    # [3] Model Training
    # =========================================================
    # model folder
    modelFolder = args.modelFolder
    if not os.path.isdir(modelFolder):
        os.makedirs(modelFolder)
    ckpointsFolder = os.path.join(modelFolder, 'checkpoints')
    if not os.path.isdir(ckpointsFolder):
        os.makedirs(ckpointsFolder)

    # sample image and segmentation used for saving template image and template segmentation
    sample_img = os.path.join(dataFolder, 'image', train_list[0])
    sample_seg = os.path.join(dataFolder, 'label', train_list[0])
    tmp = nib.load(sample_img)
    sample_array = tmp.get_fdata()

    # model construction
    imgSize = np.flip(sample_array.shape)
    model = SVF_resid(img_sz=imgSize, device=device)

    # load the saved model
    if args.cont:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.weights_init()

    # send model to device
    model.to(device)

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if args.cont:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # training
    model.train()

    if not args.cont:
        now = datetime.datetime.now()
        now_date = "{:02d}{:02d}{:02d}".format(now.month, now.day, now.year)
        now_time = "{:02d}{:02d}{:02d}".format(now.hour, now.minute, now.second)
        writer = SummaryWriter(os.path.join(modelFolder, 'logs', now_date + now_time))
    else:
        writer = SummaryWriter(args.logFile)

    # get initial template and corresponding segmentation
    if not args.cont:
        template_img_array = 0
        template_prob_array = 0
        for subID in train_list:
            # template image
            tr_img = sitk.ReadImage(os.path.join(dataFolder, 'image', subID))
            tr_img_array = np.float32(sitk.GetArrayFromImage(tr_img))
            template_img_array += tr_img_array
            # template segmentation
            tr_seg = sitk.ReadImage(os.path.join(dataFolder, 'label', subID))
            tr_seg_array = sitk.GetArrayFromImage(tr_seg)
            n_classes = np.count_nonzero(np.unique(tr_seg_array)) + 1
            mask = torch.from_numpy(tr_seg_array).unsqueeze(0)
            one_hot_shape = list(mask.shape)
            one_hot_shape[0] = n_classes
            tr_seg_one_hot = torch.zeros(one_hot_shape).scatter_(0, mask.long(), 1)
            template_prob_array += tr_seg_one_hot
        template_img_array = template_img_array / len(train_list)
        template_tensor = torch.from_numpy(template_img_array).unsqueeze(0).unsqueeze(0)
        template_prob_array = template_prob_array / len(train_list)
        template_prob = template_prob_array[np.newaxis, :, :, :, :]
        # save files
        save_init_template_img_name = os.path.join(modelFolder, 'init_template_img.nii.gz')
        save_init_template_prob_name = os.path.join(modelFolder, 'init_template_prob.pt')
        sample_img_nii = sitk.ReadImage(sample_img)
        init_template_img_nii = sitk.GetImageFromArray(template_img_array.astype('float32'))
        init_template_img_nii.CopyInformation(sample_img_nii)
        sitk.WriteImage(init_template_img_nii, save_init_template_img_name)
        torch.save(template_prob, save_init_template_prob_name)
    else:
        # load the template image
        template_img_nii = sitk.ReadImage(args.templateImg)
        template_img_array = sitk.GetArrayFromImage(template_img_nii)
        template_tensor = torch.from_numpy(template_img_array).unsqueeze(0).unsqueeze(0)
        # load the tempalte segmentation
        template_prob = torch.load(args.templateProb, map_location=device)

    # spatial transformation
    bilinear = Bilinear(zero_boundary=False)

    # identity map
    train_identity_map = gen_identity_map(imgSize).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device)
    val_identity_map = gen_identity_map(imgSize).unsqueeze(0).to(device)

    # training
    for epoch in range(epoch_start, max_epoch):
        template_imgs = template_tensor.repeat(batch_size, 1, 1, 1, 1).to(device)
        template_probs = template_prob.repeat(batch_size, 1, 1, 1, 1).to(device)
        loss_epochSum = 0
        for i, (src_imgs, src_segs, src_ids) in enumerate(dataloader_tr):
            src_imgs, src_segs = src_imgs.to(device), src_segs.to(device)
            optimizer.zero_grad()

            # get deformation field
            inputs = torch.cat((template_imgs, src_imgs), 1)
            pos_flow, neg_flow = model(inputs)
            pos_deform_field = pos_flow + train_identity_map
            neg_deform_field = neg_flow + train_identity_map

            # image registration (template to input image)
            svf_warped_template_imgs = bilinear(template_imgs, pos_deform_field)
            # image registration (input image to template)
            svf_warped_src_imgs = bilinear(src_imgs, neg_deform_field)

            # [Loss]
            # the image similarity loss
            sim_loss = get_sim_loss(svf_warped_template_imgs, src_imgs, loss_name)
            # the regularization loss
            reg_loss = get_reg_loss(pos_flow)
            # the image loss in template space
            if template_pair_sim_factor != 0.0:
                template_pair_sim_loss = get_pair_sim_loss(svf_warped_src_imgs, loss_name)
            # the image loss in image space
            if image_pair_sim_factor != 0.0:
                sec_pos_deform_field = torch.flip(pos_deform_field, dims=[0])
                sec_src_imgs = torch.flip(src_imgs, dims=[0])
                svf_warped_src_imgs_in_image_space = bilinear(src_imgs, (
                        bilinear(neg_flow, sec_pos_deform_field) + sec_pos_deform_field))
                image_pair_sim_loss = get_pair_sim_loss_image_space(svf_warped_src_imgs_in_image_space, sec_src_imgs,
                                                                    loss_name)
            # total loss
            if template_pair_sim_factor == 0.0 and image_pair_sim_factor == 0.0:
                loss = sim_factor * sim_loss + reg_factor * reg_loss
            elif template_pair_sim_factor != 0.0 and image_pair_sim_factor == 0.0:
                loss = sim_factor * sim_loss + reg_factor * reg_loss + template_pair_sim_factor * template_pair_sim_loss
            elif template_pair_sim_factor == 0.0 and image_pair_sim_factor != 0.0:
                loss = sim_factor * sim_loss + reg_factor * reg_loss + image_pair_sim_factor * image_pair_sim_loss
            elif template_pair_sim_factor != 0.0 and image_pair_sim_factor != 0.0:
                loss = sim_factor * sim_loss + reg_factor * reg_loss + template_pair_sim_factor * template_pair_sim_loss \
                       + image_pair_sim_factor * image_pair_sim_loss
            loss_epochSum += loss

            # back-propagation
            loss.backward()
            optimizer.step()

            if template_pair_sim_factor == 0.0 and image_pair_sim_factor == 0.0:
                print(
                    '[Training] epoch: {}/{}, iter: {}, total loss: {}, sim_loss: {}, reg_loss: {}'.format(
                        epoch + 1, max_epoch, i + 1, loss.item(), sim_loss.item(), reg_loss.item())
                )
            elif template_pair_sim_factor != 0.0 and image_pair_sim_factor == 0.0:
                print(
                    '[Training] epoch: {}/{}, iter: {}, total loss: {}, sim_loss: {}, reg_loss: {}, template_pair_sim_loss: {}'.format(
                        epoch + 1, max_epoch, i + 1, loss.item(), sim_loss.item(), reg_loss.item(),
                        template_pair_sim_loss.item())
                )
            elif template_pair_sim_factor == 0.0 and image_pair_sim_factor != 0.0:
                print(
                    '[Training] epoch: {}/{}, iter: {}, total loss: {}, sim_loss: {}, reg_loss: {}, image_pair_sim_loss: {}'.format(
                        epoch + 1, max_epoch, i + 1, loss.item(), sim_loss.item(), reg_loss.item(),
                        image_pair_sim_loss.item())
                )
            elif template_pair_sim_factor != 0.0 and image_pair_sim_factor != 0.0:
                print(
                    '[Training] epoch: {}/{}, iter: {}, total loss: {}, sim_loss: {}, reg_loss: {}, template_pair_sim_loss: {}, image_pair_sim_loss: {}'.format(
                        epoch + 1, max_epoch, i + 1, loss.item(), sim_loss.item(), reg_loss.item(),
                        template_pair_sim_loss.item(),
                        image_pair_sim_loss.item())
                )
            del svf_warped_template_imgs, pos_flow, pos_deform_field, neg_flow, neg_deform_field

        # log training loss
        loss_epochAvg = loss_epochSum / (i+1)
        writer.add_scalar('loss/training', loss_epochAvg, epoch + 1)

        # Validate to save the best template and model parameters
        if (epoch + 1) % args.save_per_epoch == 0:
            with torch.set_grad_enabled(False):
                # create average template segmentation
                tmp_img, tmp_seg = 0, 0
                dice_all = 0
                template_imgs = template_tensor.to(device)
                template_probs = template_prob.to(device)
                for _, (mean_src_imgs, mean_src_segs, _) in enumerate(dataloader_val):
                    mean_src_imgs, mean_src_segs = mean_src_imgs.to(device), mean_src_segs.to(device)
                    src_inputs = torch.cat((template_imgs, mean_src_imgs), 1)
                    _, mean_neg_flow_src = model(src_inputs)
                    mean_neg_deform_field_src = mean_neg_flow_src + val_identity_map
                    mean_warped_src_segs = bilinear(mean_src_segs, mean_neg_deform_field_src)
                    tmp_seg += mean_warped_src_segs
                mean_template_seg_tensor = tmp_seg / len(dataloader_val)
                del mean_src_imgs, mean_src_segs, src_inputs, mean_neg_flow_src, mean_warped_src_segs, tmp_seg

                # get the average Dice
                for _, (inf_src_imgs, inf_src_segs, _) in enumerate(dataloader_val):
                    n_classes = np.count_nonzero(np.unique(inf_src_segs)) + 1
                    inf_src_imgs, inf_src_segs = inf_src_imgs.to(device), inf_src_segs.to(device)
                    src_inputs = torch.cat((template_imgs, inf_src_imgs), 1)
                    pos_flow, _ = model(src_inputs)
                    pos_deform_field = pos_flow + val_identity_map
                    svf_warped_template_segs = bilinear(mean_template_seg_tensor, pos_deform_field)
                    dice_all += (1.0 - get_atlas_seg_loss(inf_src_segs, svf_warped_template_segs, n_classes,
                                                          flag_noBG=True))
                dice_avg = dice_all / len(dataloader_val)
                del inf_src_imgs, inf_src_segs, src_inputs, pos_deform_field, mean_template_seg_tensor, \
                    svf_warped_template_segs, dice_all

                # log the average Dice
                print("[Validation] epoch: {}/{}, val Dice: {:.5f}".format(epoch + 1, max_epoch, dice_avg))
                writer.add_scalar('validation/avgDice', dice_avg, epoch + 1)

                # save the latest model and template
                save_last_model_path = os.path.join(ckpointsFolder, 'model_last.pth.tar')
                current_state = {'epoch': epoch + 1,
                                 'state_dict': model.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'best_avgDice': best_avgDice
                                 }
                torch.save(current_state, save_last_model_path)

                # save best model
                if dice_avg > best_avgDice:
                    best_avgDice = dice_avg.item()

                    # log the best average Dice
                    print('[Best Model] epoch: {}, (current highest) Dice: {:.5f}'.format(epoch + 1, dice_avg))
                    writer.add_scalar('validation/highest_dice', dice_avg, epoch + 1)

                    # save the best model
                    save_best_model_path = os.path.join(modelFolder, 'model_best.pth.tar')
                    best_state = {'epoch': epoch + 1,
                                  'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'best_avgDice': best_avgDice
                                  }
                    torch.save(best_state, save_best_model_path)

                    # update the template image, template segmentation, and probability map
                    tmp_img, tmp_seg, JD_denominator = 0, 0, 0
                    for _, (update_src_imgs, update_src_segs, _) in enumerate(dataloader_template_update):
                        update_src_imgs, update_src_segs = update_src_imgs.to(device), update_src_segs.cuda(
                            args.gpu)
                        src_inputs = torch.cat((template_imgs, update_src_imgs), 1)
                        update_pos_flow_src, update_neg_flow_src = model(src_inputs)
                        update_pos_deform_field_src = update_pos_flow_src + val_identity_map
                        update_neg_deform_field_src = update_neg_flow_src + val_identity_map
                        update_warped_src_imgs = bilinear(update_src_imgs, update_neg_deform_field_src)
                        update_warped_src_segs = bilinear(update_src_segs, update_neg_deform_field_src)
                        JD_tensor = torch.from_numpy(jacobian_determinant(update_neg_deform_field_src)).unsqueeze(
                            0).unsqueeze(0).to(device)
                        JD_denominator += JD_tensor
                        tmp_img += (update_warped_src_imgs * JD_tensor)
                        tmp_seg += (update_warped_src_segs * JD_tensor)
                    template_tensor = tmp_img / JD_denominator
                    template_tensor.clamp_(min=0.0, max=1.0)
                    template_prob = tmp_seg / JD_denominator
                    # save files (checkpoints)
                    dir_ckpoints_img = os.path.join(ckpointsFolder, 'template_img')
                    if not os.path.isdir(dir_ckpoints_img):
                        os.makedirs(dir_ckpoints_img)
                    dir_ckpoints_seg = os.path.join(ckpointsFolder, 'template_seg')
                    if not os.path.isdir(dir_ckpoints_seg):
                        os.makedirs(dir_ckpoints_seg)
                    dir_ckpoints_prob = os.path.join(ckpointsFolder, 'template_prob')
                    if not os.path.isdir(dir_ckpoints_prob):
                        os.makedirs(dir_ckpoints_prob)
                    save_ckpoints_template_img_name = os.path.join(dir_ckpoints_img,
                                                                   'epoch' + str(epoch + 1) + '_template_img.nii.gz')
                    save_ckpoints_template_seg_name = os.path.join(dir_ckpoints_seg,
                                                                   'epoch' + str(epoch + 1) + '_template_seg.nii.gz')
                    save_ckpoints_template_prob_name = os.path.join(dir_ckpoints_prob,
                                                                    'epoch' + str(epoch + 1) + '_template_prob.pt')
                    save_updated_atlas(template_tensor, template_prob, save_ckpoints_template_img_name,
                                       save_ckpoints_template_seg_name,
                                       save_ckpoints_template_prob_name, sample_img, sample_seg)
                    # save files (best model)
                    save_template_img_name = os.path.join(modelFolder, 'template_img.nii.gz')
                    save_template_seg_name = os.path.join(modelFolder, 'template_seg.nii.gz')
                    save_template_prob_name = os.path.join(modelFolder, 'template_prob.pt')
                    save_updated_atlas(template_tensor, template_prob, save_template_img_name, save_template_seg_name,
                                       save_template_prob_name, sample_img, sample_seg)
                    del update_src_imgs, update_src_segs, src_inputs, update_pos_flow_src, update_neg_flow_src, \
                        update_pos_deform_field_src, update_neg_deform_field_src, update_warped_src_imgs, \
                        update_warped_src_segs, JD_tensor, JD_denominator, tmp_img, tmp_seg

    writer.close()
    # =========================================================
