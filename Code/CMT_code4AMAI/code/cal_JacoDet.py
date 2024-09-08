import os
import scipy
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import csv
import argparse


def cal_jacobian_determinant_V1(deformation_field):
    """
    Calculate the Jacobian determinant of a deformation field.
    
    :param deformation_field: numpy array of shape [x, y, z, 3]
    :return: Jacobian determinant field of shape [x, y, z]
    """
    
    # JacobianMatrix: [dxx, dxy, dxz],[dyx,dyy,dyz],[dzx,dzy,dzz]]
    dxx = np.gradient(deformation_field[..., 0], axis=0)
    dyy = np.gradient(deformation_field[..., 1], axis=1)
    dzz = np.gradient(deformation_field[..., 2], axis=2)
    dxy = np.gradient(deformation_field[..., 0], axis=1)
    dyz = np.gradient(deformation_field[..., 1], axis=2)
    dzx = np.gradient(deformation_field[..., 2], axis=0)
    dxz = np.gradient(deformation_field[..., 0], axis=2)
    dyx = np.gradient(deformation_field[..., 1], axis=0)
    dzy = np.gradient(deformation_field[..., 2], axis=1)
    dxz = np.gradient(deformation_field[..., 0], axis=2)
    dzx = np.gradient(deformation_field[..., 2], axis=0)
    dxy = np.gradient(deformation_field[..., 0], axis=1)
    dyx = np.gradient(deformation_field[..., 1], axis=0)
    dyz = np.gradient(deformation_field[..., 1], axis=2)
    dzy = np.gradient(deformation_field[..., 2], axis=1)
    
    # Calculate Jacobian determinant 
    jacobian_det = (dxx) * (dyy) * (dzz) + dxy * dyz * dzx + dxz * dyx * dzy - dxz * (dyy) * dzx - dxy * dyx * (dzz) - (dxx) * dyz * dzy   
    
    return jacobian_det



# adapted from LapIRN: 
# https://github.com/cwmok/LapIRN/blob/549a40329d4e8304e35c6f0113c46ad8d7a6da32/Code/miccai2020_model_stage.py#L782C4-L782C29
def cal_jacobian_determinant_V2(deformation_field):
    """
    Calculate the Jacobian determinant of a deformation field.
    
    :param deformation_field: numpy array of shape [x, y, z, 3]
    :return: Jacobian determinant field of shape [x, y, z]
    """

    dx = deformation_field[1:, :-1, :-1, :] - deformation_field[:-1, :-1, :-1, :]
    dy = deformation_field[:-1, 1:, :-1, :] - deformation_field[:-1, :-1, :-1, :]
    dz = deformation_field[:-1, :-1, 1:, :] - deformation_field[:-1, :-1, :-1, :]

    # JacobianMatrix: [dxx, dxy, dxz],[dyx,dyy,dyz],[dzx,dzy,dzz]]
    Jdet0 = dx[:,:,:,0] * (dy[:,:,:,1] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,1]) # this is dxx * (dyy*dzz - dzy*dyz)
    Jdet1 = dx[:, :,:,1] * (dy[:,:,:,0] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,0]) # this is dyx * (dxy*dzz - dzy*dxz)
    Jdet2 = dx[:,:,:,2] * (dy[:,:,:,0] * dz[:,:,:,1] - dy[:,:,:,1] * dz[:,:,:,0]) # this is dzx * (dxy*dyz - dyy*dxz)
    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet



# adapted from L2R:
# https://github.com/MDL-UzL/L2R/blob/fc713f5baf107932a8f72ab55484bf7ac3210b57/evaluation/utils.py#L29
def cal_jacobian_determinant_V3(deformation_field):
    """
    Calculate the Jacobian determinant of a deformation field.
    
    :param deformation_field: numpy array of shape [x, y, z, 3]
    :return: Jacobian determinant field of shape [x, y, z]
    """ 

    # Create identity grid
    grid = np.stack(np.meshgrid(np.arange(deformation_field.shape[0]),
                                    np.arange(deformation_field.shape[1]),
                                    np.arange(deformation_field.shape[2]), 
                                    indexing='ij'), axis=-1).astype(np.float32)
    # Calculate displacement field
    disp = deformation_field - grid

    disp = np.transpose(disp, (3, 0, 1, 2)) # change dimensions to [3, x, y, z] 
    disp = disp[None, ...] # change dimensions to [1, 3, x, y, z] 
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
        
    return jacdet



def test_jacobian_determinant_V3():
    # testing data: 
    # deformation_field: numpy array of shape [x, y, z, 3] 

    # Test case 1: Identity transformation
    print("Test case 1: Identity transformation should have Jacobian determinant of 1")
    identity = np.stack(np.meshgrid(np.arange(10), np.arange(10), np.arange(10), indexing='ij'), axis=-1).astype(np.float32)
    jac_det = cal_jacobian_determinant_V3(identity)
    assert np.allclose(jac_det, 1.0), "Identity transformation should have Jacobian determinant of 1"

    # Test case 2: Uniform scaling
    scale = 2.0
    print(f"Test case 2: Uniform scaling by {scale} should have Jacobian determinant of {scale**3}")
    scaled = identity * scale
    jac_det = cal_jacobian_determinant_V3(scaled)
    assert np.allclose(jac_det, scale**3), f"Uniform scaling by {scale} should have Jacobian determinant of {scale**3}"

    # Test case 3: Translation
    print("Test case 3: Pure translation should have Jacobian determinant of 1")
    translated = identity + np.array([1.0, 2.0, 3.0])
    jac_det = cal_jacobian_determinant_V3(translated)
    assert np.allclose(jac_det, 1.0), "Pure translation should have Jacobian determinant of 1"

    # Test case 4: Simple shear
    print("Test case 4: Simple shear should have Jacobian determinant of 1")
    shear = identity.copy()
    shear[..., 0] += 0.1 * identity[..., 1]
    jac_det = cal_jacobian_determinant_V3(shear)
    assert np.allclose(jac_det, 1.0), "Simple shear should have Jacobian determinant of 1"

    # Test case 5: Compression in one direction
    print("Test case 5: Compression by 0.5 in one direction should have Jacobian determinant of 0.5")
    compressed = identity.copy()
    compressed[..., 0] *= 0.5
    jac_det = cal_jacobian_determinant_V3(compressed)
    assert np.allclose(jac_det, 0.5), "Compression by 0.5 in one direction should have Jacobian determinant of 0.5"

    print("All tests passed!")



def test_jacobian_determinant_V2():
    # testing data: 
    # deformation_field: numpy array of shape [x, y, z, 3] 

    # Test case 1: Identity transformation
    print("Test case 1: Identity transformation should have Jacobian determinant of 1")
    identity = np.stack(np.meshgrid(np.arange(10), np.arange(10), np.arange(10), indexing='ij'), axis=-1).astype(np.float32)
    jac_det = cal_jacobian_determinant_V2(identity)
    assert np.allclose(jac_det, 1.0), "Identity transformation should have Jacobian determinant of 1"

    # Test case 2: Uniform scaling
    scale = 2.0
    print(f"Test case 2: Uniform scaling by {scale} should have Jacobian determinant of {scale**3}")
    scaled = identity * scale
    jac_det = cal_jacobian_determinant_V2(scaled)
    assert np.allclose(jac_det, scale**3), f"Uniform scaling by {scale} should have Jacobian determinant of {scale**3}"

    # Test case 3: Translation
    print("Test case 3: Pure translation should have Jacobian determinant of 1")
    translated = identity + np.array([1.0, 2.0, 3.0])
    jac_det = cal_jacobian_determinant_V2(translated)
    assert np.allclose(jac_det, 1.0), "Pure translation should have Jacobian determinant of 1"

    # Test case 4: Simple shear
    print("Test case 4: Simple shear should have Jacobian determinant of 1")
    shear = identity.copy()
    shear[..., 0] += 0.1 * identity[..., 1]
    jac_det = cal_jacobian_determinant_V2(shear)
    assert np.allclose(jac_det, 1.0), "Simple shear should have Jacobian determinant of 1"

    # Test case 5: Compression in one direction
    print("Test case 5: Compression by 0.5 in one direction should have Jacobian determinant of 0.5")
    compressed = identity.copy()
    compressed[..., 0] *= 0.5
    jac_det = cal_jacobian_determinant_V2(compressed)
    assert np.allclose(jac_det, 0.5), "Compression by 0.5 in one direction should have Jacobian determinant of 0.5"

    print("All tests passed!")



def test_jacobian_determinant_V1():
    # testing data: 
    # deformation_field: numpy array of shape [x, y, z, 3] 

    # Test case 1: Identity transformation
    print("Test case 1: Identity transformation should have Jacobian determinant of 1")
    identity = np.stack(np.meshgrid(np.arange(10), np.arange(10), np.arange(10), indexing='ij'), axis=-1).astype(np.float32)
    jac_det = cal_jacobian_determinant_V1(identity)
    assert np.allclose(jac_det, 1.0), "Identity transformation should have Jacobian determinant of 1"

    # Test case 2: Uniform scaling
    scale = 2.0
    print(f"Test case 2: Uniform scaling by {scale} should have Jacobian determinant of {scale**3}")
    scaled = identity * scale
    jac_det = cal_jacobian_determinant_V1(scaled)
    assert np.allclose(jac_det, scale**3), f"Uniform scaling by {scale} should have Jacobian determinant of {scale**3}"

    # Test case 3: Translation
    print("Test case 3: Pure translation should have Jacobian determinant of 1")
    translated = identity + np.array([1.0, 2.0, 3.0])
    jac_det = cal_jacobian_determinant_V1(translated)
    assert np.allclose(jac_det, 1.0), "Pure translation should have Jacobian determinant of 1"

    # Test case 4: Simple shear
    print("Test case 4: Simple shear should have Jacobian determinant of 1")
    shear = identity.copy()
    shear[..., 0] += 0.1 * identity[..., 1]
    jac_det = cal_jacobian_determinant_V1(shear)
    assert np.allclose(jac_det, 1.0), "Simple shear should have Jacobian determinant of 1"

    # Test case 5: Compression in one direction
    print("Test case 5: Compression by 0.5 in one direction should have Jacobian determinant of 0.5")
    compressed = identity.copy()
    compressed[..., 0] *= 0.5
    jac_det = cal_jacobian_determinant_V1(compressed)
    assert np.allclose(jac_det, 0.5), "Compression by 0.5 in one direction should have Jacobian determinant of 0.5"

    print("All tests passed!")



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate Jacobian determinant of the deformation field of size [d1,d2,d3,3]')
    parser.add_argument('--test', action='store_true', help='testing my mode')
    parser.add_argument('--test_L2R', action='store_true', help='testing code from L2R')
    parser.add_argument('--test_LapIRN', action='store_true', help='testing code from LapIRN')
    parser.add_argument('--deformationField_dir', type=str, help='Path to directory containing deformation fields')
    parser.add_argument('--eval_dir', type=str, help='Path to directory where evaluation results should be saved')
    args = parser.parse_args()

    if args.test:
        print("Testing my code")
        test_jacobian_determinant_V1()
    elif args.test_LapIRN:
        print(f"Testing the code adapted from LapIRN\nhttps://github.com/cwmok/LapIRN/blob/549a40329d4e8304e35c6f0113c46ad8d7a6da32/Code/miccai2020_model_stage.py#L782C4-L782C29")
        test_jacobian_determinant_V2()
    elif args.test_L2R:
        print(f"Testing the code adapted from L2R\nhttps://github.com/MDL-UzL/L2R/blob/fc713f5baf107932a8f72ab55484bf7ac3210b57/evaluation/utils.py#L29")
        test_jacobian_determinant_V3()
    else:
        # Create evaluation directory if it doesn't exist
        if not os.path.exists(args.eval_dir):
            os.makedirs(args.eval_dir)

        # Get list of NIfTI files
        field_files = [f for f in os.listdir(args.deformationField_dir) if f.endswith('.nii.gz')]

        # Initialize results list
        JacoDet_results = []

        # Loop over field_files
        for file in field_files:
            # Read NIfTI file
            field_nii = sitk.ReadImage(os.path.join(args.deformationField_dir, file))
            field_data = sitk.GetArrayFromImage(field_nii).transpose([3,1,2,0])
            # JacoDet
            JacoDet = cal_jacobian_determinant_V1(field_data)
            # std(JacoDet)
            SDJacoDet = np.std(JacoDet)
            # std(log(JacoDet))
            JacoDet_nonzero = np.where(JacoDet == 0, np.finfo(float).eps, JacoDet)
            SDlogJacoDet = np.std(np.log(np.abs(JacoDet_nonzero)))
            # percentage_foldings
            percentage_foldings = np.sum(JacoDet <= 0) / JacoDet.size
            # all metrics
            JacoDet_results.append([file, SDJacoDet, SDlogJacoDet, percentage_foldings])

        # Save results to CSV file
        with open(os.path.join(args.eval_dir, 'eval_JacoDet.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File', 'SDJacoDet', 'SDlogJacoDet', 'percentage_foldings'])
            for result in JacoDet_results:
                writer.writerow(result)



if __name__ == "__main__":
    main()
