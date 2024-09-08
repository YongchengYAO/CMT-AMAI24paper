# modified from SVF_test_forward_atlas_example.py

import torch
import numpy as np
from datasets import *
from torch.utils.data import DataLoader
import evalMetrics as metrics
import os
import sys
from atlas_models import SVF_resid
import SimpleITK as sitk
from atlas_utils import *
from viz_deformation import *
sys.path.append(os.path.realpath(".."))
import argparse
import glob
import matplotlib.pyplot as plt
import nibabel as nib


parser = argparse.ArgumentParser()

# gpu
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

# folder
parser.add_argument('--dataFolder', type=str, help='data folder')
parser.add_argument('--modelFolder', type=str, help='model folder')
parser.add_argument('--warpingFieldFolder', type=str, help='deformation field')
parser.add_argument('--warpedTempImgFolder', type=str, help='folder for the warped template image')
parser.add_argument('--warpedTempSegFolder', type=str, help='folder for the warped template mask')

args = parser.parse_args()


if __name__ == "__main__":

    # device
    if args.gpu is not None:
        device = torch.device('cuda', args.gpu)
    elif args.mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # get subject list of testing set
    dataFolder = args.dataFolder
    listPath_ts = os.path.join(dataFolder, 'test.txt')
    test_list = getSubID_Ts(listPath_ts)

    # dataloader for the testing set
    SVFNet_test_single = OAIZIB_img_Vt(dataFolder, test_list)
    SVFNet_test_single_dataloader = DataLoader(SVFNet_test_single, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)

    # atla files
    modelFolder = args.modelFolder
    atlas_img_file = os.path.join(modelFolder, 'template_img.nii.gz')
    atlas_seg_file = os.path.join(modelFolder, 'template_seg.nii.gz')

    # atlas image
    atlas_img = sitk.ReadImage(atlas_img_file)
    atlas_img_array = sitk.GetArrayFromImage(atlas_img)
    atlas_img_tensor = torch.from_numpy(atlas_img_array).unsqueeze(0).unsqueeze(0).to(device)

    # atals segmentation mask
    atlas_seg = sitk.ReadImage(atlas_seg_file)
    atlas_seg_array = sitk.GetArrayFromImage(atlas_seg)
    atlas_seg_tensor = torch.from_numpy(atlas_seg_array).unsqueeze(0).unsqueeze(0).to(device)

    # model file
    model_file = os.path.join(modelFolder, 'model_best.pth.tar')

    # model input size
    imgSize = atlas_img_array.shape # compatiable with the model input size convention in Aladdin_OAIZIB_train.py

    ## load model
    svf_model_state_dict = torch.load(model_file, map_location=device)
    svf_model = SVF_resid(img_sz=imgSize, device=device)
    torch.cuda.set_device(device)
    svf_model.to(device)
    svf_model.load_state_dict(svf_model_state_dict['state_dict'], strict=False)
    svf_model.eval()

    # spatial transformation
    bilinear = Bilinear(zero_boundary=True)
    nearestNeighbour = NearestNeighbour(zero_boundary=True)

    dice_all_atlas, dice_all_image = 0, 0
    identity_map = gen_identity_map(imgSize).unsqueeze(0).to(device)
    tmp_img, tmp_seg = 0, 0

    # warp template mask to image space
    with torch.set_grad_enabled(False):
        for j, (target_img_tensor, filename) in enumerate(SVFNet_test_single_dataloader):
            # target image
            target_img_tensor = target_img_tensor.to(device)

            # model input
            src_cat_input = torch.cat((atlas_img_tensor, target_img_tensor), 1)
            mean_pos_flow_src, _ = svf_model(src_cat_input)

            # deformation field: 
            # [1] positive field: template to image
            # [2] negative field: image to template
            warpingField_template2img_tensor = mean_pos_flow_src + identity_map

            # warp template image and mask to image space
            warped_template_img_tensor = bilinear(atlas_img_tensor, warpingField_template2img_tensor)
            warped_template_seg_tensor = nearestNeighbour(atlas_seg_tensor, warpingField_template2img_tensor)

            # load target image nii
            target_img_nii = sitk.ReadImage(os.path.join(dataFolder, 'image', filename[0]))

            # save the template-to-image deformation field
            save_warping_field_name = os.path.join(args.warpingFieldFolder, filename[0])
            warpingField_template2img_np = warpingField_template2img_tensor.detach().squeeze().cpu().numpy().transpose([1,2,3,0])
            warpingField_template2img_nii = sitk.GetImageFromArray(warpingField_template2img_np.astype('float32'))
            warpingField_template2img_nii.CopyInformation(target_img_nii)
            sitk.WriteImage(warpingField_template2img_nii, save_warping_field_name)

            # save warped template image
            save_warped_template_img_name = os.path.join(args.warpedTempImgFolder, filename[0])
            warped_template_img_np = warped_template_img_tensor.detach().squeeze().cpu().numpy()
            warped_template_img_nii = sitk.GetImageFromArray(warped_template_img_np.astype('float32'))
            warped_template_img_nii.CopyInformation(target_img_nii)
            sitk.WriteImage(warped_template_img_nii, save_warped_template_img_name)

            # save warped template mask
            save_warped_template_seg_name = os.path.join(args.warpedTempSegFolder, filename[0])
            warped_template_seg_np = warped_template_seg_tensor.detach().squeeze().cpu().numpy()
            warped_template_seg_nii = sitk.GetImageFromArray(warped_template_seg_np.astype('float32'))
            warped_template_seg_nii.CopyInformation(target_img_nii)
            sitk.WriteImage(warped_template_seg_nii, save_warped_template_seg_name)
