import os
import nibabel as nib
import numpy as np
import csv
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate Dice Similarity Coefficient between model predictions and ground truth segmentations.')
    parser.add_argument('--labels', nargs='+', type=int, default=[1,2,3,4,5], help='labels for ROIs to be evaluated')
    parser.add_argument('--prediction_dir', type=str, help='Path to directory containing model prediction NIfTI files')
    parser.add_argument('--GT_dir', type=str, help='Path to directory containing ground truth segmentation NIfTI files')
    parser.add_argument('--eval_dir', type=str, help='Path to directory where evaluation results should be saved')
    args = parser.parse_args()


    # Create evaluation directory if it doesn't exist
    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir)

    # Define labels to calculate DSC for
    labels = np.array(args.labels)

    # Get list of NIfTI files in prediction directory
    prediction_files = [f for f in os.listdir(args.prediction_dir) if f.endswith('.nii.gz')]

    # Initialize DSC results list
    dsc_results = []

    # Loop over prediction files
    for file in prediction_files:
        # Read prediction NIfTI file
        pred_nii = nib.load(os.path.join(args.prediction_dir, file))
        pred_data = pred_nii.get_fdata()
        # Read ground truth NIfTI file
        gt_nii = nib.load(os.path.join(args.GT_dir, file))
        gt_data = gt_nii.get_fdata()
        # Calculate DSC for each label
        for _, label in enumerate(labels):
            pred_mask = (pred_data == label)
            gt_mask = (gt_data == label)
            numerator = np.sum(np.logical_and(pred_mask, gt_mask)) * 2.0
            denominator = np.sum(pred_mask) + np.sum(gt_mask)
            if denominator != 0:
                dsc = numerator / denominator
                # Save DSC result
                dsc_results.append([file, label, dsc])

    # Save DSC results to CSV file
    with open(os.path.join(args.eval_dir, 'eval_DSC.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File', 'Label', 'DSC'])
        for result in dsc_results:
            writer.writerow(result)


if __name__ == "__main__":
    main()