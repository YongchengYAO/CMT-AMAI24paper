import os
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import csv
import argparse
import pytest

def cal_jacobian_determinant(deformation_field):
    """
    Calculate the Jacobian determinant of a deformation field.
    
    :param deformation_field: numpy array of shape [depth, height, width, 3]
    :return: Jacobian determinant field of shape [depth, height, width]
    """
    # Create identity grid
    identity = np.stack(np.meshgrid(np.arange(deformation_field.shape[0]),
                                    np.arange(deformation_field.shape[1]),
                                    np.arange(deformation_field.shape[2]),
                                    indexing='ij'), axis=-1).astype(np.float32)

    # test
    displacement_field = deformation_field - identity

    # Get gradients of displacement field
    grad_x = np.gradient(displacement_field[..., 0], axis=0)
    grad_y = np.gradient(displacement_field[..., 1], axis=1)
    grad_z = np.gradient(displacement_field[..., 2], axis=2)

    # Calculate Jacobian determinant
    jacobian_det = (1 + grad_x) * (1 + grad_y) * (1 + grad_z) + \
                   np.gradient(displacement_field[..., 1], axis=0) * \
                   np.gradient(displacement_field[..., 2], axis=1) * \
                   np.gradient(displacement_field[..., 0], axis=2) + \
                   np.gradient(displacement_field[..., 2], axis=0) * \
                   np.gradient(displacement_field[..., 0], axis=1) * \
                   np.gradient(displacement_field[..., 1], axis=2) - \
                   np.gradient(displacement_field[..., 2], axis=0) * \
                   (1 + grad_y) * np.gradient(displacement_field[..., 0], axis=2) - \
                   np.gradient(displacement_field[..., 1], axis=0) * \
                   np.gradient(displacement_field[..., 0], axis=1) * (1 + grad_z) - \
                   (1 + grad_x) * np.gradient(displacement_field[..., 2], axis=1) * \
                   np.gradient(displacement_field[..., 1], axis=2)

    return jacobian_det



# Usage:
#   run test_jacobian_determinant()
def test_jacobian_determinant():
    # Test case 1: Identity transformation
    identity = np.stack(np.meshgrid(np.arange(10), np.arange(10), np.arange(10), indexing='ij'), axis=-1).astype(np.float32)
    jac_det = cal_jacobian_determinant(identity)
    assert np.allclose(jac_det, 1.0), "Identity transformation should have Jacobian determinant of 1"

    # Test case 2: Uniform scaling
    scale = 2.0
    scaled = identity * scale
    jac_det = cal_jacobian_determinant(scaled)
    assert np.allclose(jac_det, scale**3), f"Uniform scaling by {scale} should have Jacobian determinant of {scale**3}"

    # Test case 3: Translation
    translated = identity + np.array([1.0, 2.0, 3.0])
    jac_det = cal_jacobian_determinant(translated)
    assert np.allclose(jac_det, 1.0), "Pure translation should have Jacobian determinant of 1"

    # Test case 4: Simple shear
    shear = identity.copy()
    shear[..., 0] += 0.1 * identity[..., 1]
    jac_det = cal_jacobian_determinant(shear)
    assert np.allclose(jac_det, 1.0), "Simple shear should have Jacobian determinant of 1"

    # Test case 6: Compression in one direction
    compressed = identity.copy()
    compressed[..., 0] *= 0.5
    jac_det = cal_jacobian_determinant(compressed)
    assert np.allclose(jac_det, 0.5), "Compression by 0.5 in one direction should have Jacobian determinant of 0.5"

    print("All tests passed!")

if __name__ == "__main__":
    test_jacobian_determinant()
