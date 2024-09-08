import argparse
import os
import nibabel as nib
import numpy as np
import pandas as pd
import gudhi as gd

def calculate_betti_numbers(segmentation):
    cubical_complex = gd.CubicalComplex(dimensions=segmentation.shape, top_dimensional_cells=segmentation.flatten())
    cubical_complex.persistence()
    betti_numbers = cubical_complex.betti_numbers()
    # Betti numbers are returned as a dictionary with Betti 0 and Betti 1
    betti_0 = betti_numbers[0] if 0 in betti_numbers else 0
    betti_1 = betti_numbers[1] if 1 in betti_numbers else 0
    return betti_0, betti_1

def process_segmentation_file(file_path, roi_labels):
    # Load the segmentation mask
    img = nib.load(file_path)
    segmentation = img.get_fdata().astype(int)

    results = []

    for roi in roi_labels:
        if roi == 0:
            continue  # Skip background
        roi_mask = (segmentation == roi).astype(int)
        betti_0, betti_1 = calculate_betti_numbers(roi_mask)
        results.append([file_path, roi, betti_0, betti_1])

    return results

def save_results_to_csv(results, output_csv):
    df = pd.DataFrame(results, columns=['Filename', 'ROI Label', 'Betti 0', 'Betti 1'])
    df.to_csv(output_csv, index=False)

def main():
    parser = argparse.ArgumentParser(description='Calculate Betti numbers for ROIs in NIfTI segmentation masks.')
    parser.add_argument('--labels', nargs='+', type=int, default=[1, 2, 3, 4, 5], help='Labels for ROIs to be evaluated')
    parser.add_argument('--prediction_dir', type=str, required=True, help='Path to directory containing NIfTI files')
    parser.add_argument('--eval_dir', type=str, required=True, help='Path to directory where evaluation results should be saved')
    args = parser.parse_args()

    all_results = []

    for filename in os.listdir(args.prediction_dir):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(args.prediction_dir, filename)
            results = process_segmentation_file(file_path, args.labels)
            all_results.extend(results)

    output_csv = os.path.join(args.eval_dir, 'eval_Betti_numbers.csv')
    save_results_to_csv(all_results, output_csv)

if __name__ == '__main__':
    main()
