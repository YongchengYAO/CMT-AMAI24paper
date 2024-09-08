import csv
from collections import defaultdict
import argparse

def calculate_BettiError(csv_path, GT_Betti0, GT_Betti1):
    # Initialize a dictionary to store the sum of BettiError and count for each label
    label_sums = defaultdict(lambda: {'sum_BE0': 0, 'count_BE0': 0, 'sum_BE1': 0, 'count_BE1': 0})

    # Read the CSV file
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header if present
        
        for row in reader:
            _, label, Betti0, Betti1 = row
            label = int(label)
            Betti0 = float(Betti0)
            Betti1 = float(Betti1)
            
            # Betti Error
            label_sums[label]['sum_BE0'] += Betti0 - GT_Betti0
            label_sums[label]['count_BE0'] += 1
            label_sums[label]['sum_BE1'] += Betti1 - GT_Betti1
            label_sums[label]['count_BE1'] += 1

    # Calculate and print the mean DSC for each label
    print("Mean Betti-0 Error for each label:")
    for label in sorted(label_sums.keys()):
        mean_BE0 = label_sums[label]['sum_BE0'] / label_sums[label]['count_BE0']
        print(f"Label {label}: {mean_BE0:.4f}")
    
    print("Mean Betti-1 Error for each label:")
    for label in sorted(label_sums.keys()):
        mean_BE1 = label_sums[label]['sum_BE1'] / label_sums[label]['count_BE1']
        print(f"Label {label}: {mean_BE1:.4f}")

    # Calculate overall mean DSC
    total_sum_BE0 = sum(data['sum_BE0'] for data in label_sums.values())
    total_count_BE0 = sum(data['count_BE0'] for data in label_sums.values())
    overall_mean_BE0 = total_sum_BE0 / total_count_BE0
    print(f"Overall Mean Betti-0 Error: {overall_mean_BE0:.4f}\n")

    total_sum_BE1 = sum(data['sum_BE1'] for data in label_sums.values())
    total_count_BE1 = sum(data['count_BE1'] for data in label_sums.values())
    overall_mean_BE1 = total_sum_BE1 / total_count_BE1
    print(f"Overall Mean Betti-1 Error: {overall_mean_BE1:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate mean DSC from CSV file.')
    parser.add_argument('--csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('--GT_Betti0', type=int, help='Ground truth of Betti-0')
    parser.add_argument('--GT_Betti1', type=int, help='Ground truth of Betti-1')
    args = parser.parse_args()

    calculate_BettiError(args.csv_path, args.GT_Betti0, args.GT_Betti1)
