import csv
import argparse
import pandas as pd

def calculate_mean_JacoDet_metrics(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Calculate the mean of 'JacoDet_std' and 'JacoDet_nonpositive'
    mean_SDJacoDet = df['SDJacoDet'].mean()
    mean_SDlogJacoDet = df['SDlogJacoDet'].mean()
    mean_percentage_foldings = df['percentage_foldings'].mean()

    print(f"Mean SDJacoDet: {mean_SDJacoDet}")
    print(f"Mean SDlogJacoDet: {mean_SDlogJacoDet}")
    print(f"Mean percentage_foldings: {mean_percentage_foldings}\n")
    
    return mean_SDJacoDet, mean_SDlogJacoDet, mean_percentage_foldings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate mean metrics from CSV file.')
    parser.add_argument('--csv_path', type=str, help='Path to the CSV file')
    args = parser.parse_args()

    calculate_mean_JacoDet_metrics(args.csv_path)
