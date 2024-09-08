import os
from argparse import ArgumentParser

import numpy as np
import torch
import nibabel as nib

from Functions import generate_grid_unit, save_img, save_flow, transform_unit_flow_to_flow, load_4D, imgnorm
from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_add_lvl1, Miccai2020_LDR_laplacian_unit_add_lvl2, \
    Miccai2020_LDR_laplacian_unit_add_lvl3, SpatialTransform_unit, SpatialTransformNearest_unit

parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='../Model/LapIRN_diff_fea7.pth',
                    help="Pre-trained Model path")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='../Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")
parser.add_argument("--fixed", type=str,
                    dest="fixed", default='../Data/image_A.nii',
                    help="fixed image")
parser.add_argument("--moving", type=str,
                    dest="moving", default='../Data/image_B.nii',
                    help="moving image")

# -----------------------
# added by YC Yao
# -----------------------
# Use nargs=3 to expect three values for the --dim argument
parser.add_argument("--dim", type=int, nargs=3, required=True, help="Dimensions of image shape (three values expected)")
parser.add_argument("--gpu", type=int)
parser.add_argument("--movingSeg", type=str, help="segmentation mask for the moving image, used for evaluation")
parser.add_argument("--movingDir", type=str,
		    help="folder of moving images")
parser.add_argument("--fixedDir", type=str,
             help="folder of fixed images")
parser.add_argument("--flowDir", type=str,
             help="folder of warpped flow")
parser.add_argument("--movedSegDir", type=str,
             help="folder of moved segmentation mask")
# -----------------------

opt = parser.parse_args()

savepath = opt.savepath
moving_path = opt.moving
if not os.path.isdir(savepath):
    os.makedirs(savepath)

# -----------------------
# added by YC Yao
# -----------------------
if not os.path.isdir(opt.flowDir):
    os.makedirs(opt.flowDir)
if not os.path.isdir(opt.movedSegDir):
    os.makedirs(opt.movedSegDir)
# -----------------------

start_channel = opt.start_channel



def test():
    # -----------------------
    # modified by YC
    # -----------------------
    imgshape = tuple(opt.dim)
    imgshape_4 = tuple([dim // 4 for dim in imgshape])
    imgshape_2 = tuple([dim // 2 for dim in imgshape])
    # -----------------------  

    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                               range_flow=range_flow).cuda()
    model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).cuda()

    model = Miccai2020_LDR_laplacian_unit_add_lvl3(2, 3, start_channel, is_train=False, imgshape=imgshape,
                                          range_flow=range_flow, model_lvl2=model_lvl2).cuda()

    transform = SpatialTransform_unit().cuda()
    model.load_state_dict(torch.load(opt.modelpath))
    model.eval()
    transform.eval()

    # -----------------------
    # added by YC Yao
    # -----------------------
    transform_nn = SpatialTransformNearest_unit().cuda()
    transform_nn.eval()
    # -----------------------

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    use_cuda = True
    device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu')


    # -----------------------
    # modified by YC Yao
    # -----------------------
    # one moving image, such as a template image
    moving_img = load_4D(moving_path)
    # normalize image to [0, 1]
    moving_img = imgnorm(moving_img)
    if isinstance(moving_img, np.ndarray):
        moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)
    elif isinstance(moving_img, torch.Tensor):
        moving_img = moving_img.float().to(device).unsqueeze(dim=0)

    # segmentation mask for the moving image
    moving_seg = load_4D(opt.movingSeg)
    if isinstance(moving_seg, np.ndarray):
        moving_seg = torch.from_numpy(moving_seg).float().to(device).unsqueeze(dim=0)
    elif isinstance(moving_seg, torch.Tensor):
        moving_seg = moving_seg.float().to(device).unsqueeze(dim=0)

    # List to store the file names
    nii_gz_files = []
    nii_gz_filenames = []

    # Walk through the directory
    for root, dirs, files in os.walk(opt.fixedDir):
        for file in files:
            if file.endswith(".nii.gz"):
                # Append full file path
                nii_gz_files.append(os.path.join(root, file))    
                nii_gz_filenames.append(file)    

    # loop over all fixed images
    for idx, i_fixed_path in enumerate(nii_gz_files):
        # load fixed image 
        img_name = nii_gz_filenames[idx]
        fixed_img = load_4D(i_fixed_path)
        fixed_img_nii = nib.load(i_fixed_path)
        # Get the affine matrix
        fixed_img_affine = fixed_img_nii.affine
        # Get the header
        fixed_img_header = fixed_img_nii.header
        

        # normalize image to [0, 1]
        fixed_img = imgnorm(fixed_img)

        # modified by YC
        # -------
        if isinstance(fixed_img, np.ndarray):
            fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
        elif isinstance(fixed_img, torch.Tensor):
            fixed_img = fixed_img.float().to(device).unsqueeze(dim=0)
        # -------

        with torch.no_grad():
            # flow
            F_X_Y = model(moving_img, fixed_img)
            
            # transform the moving image
            X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]

            # added by YC
            # transform the moving segmentation mask
            Xseg_Y = transform_nn(moving_seg, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]

            F_X_Y_cpu = F_X_Y.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

            save_flow(F_X_Y_cpu, opt.flowDir+'/'+img_name, header=fixed_img_header, affine=fixed_img_affine)
            save_img(X_Y, savepath+'/'+img_name, header=fixed_img_header, affine=fixed_img_affine)

            # added by YC
            # save the moved mask
            save_img(Xseg_Y, opt.movedSegDir+'/'+img_name, header=fixed_img_header, affine=fixed_img_affine)


    print("Finished")
    # -----------------------



if __name__ == '__main__':
    range_flow = 0.4
    test()
