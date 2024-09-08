import csv
from collections import defaultdict
import argparse

def calculate_mean_dsc(csv_path):
    # Initialize a dictionary to store the sum of DSC values and count for each label
    label_sums = defaultdict(lambda: {'sum': 0, 'count': 0})

    # Read the CSV file
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header if present
        
        for row in reader:
            _, label, dsc = row
            label = int(label)
            dsc = float(dsc)
            
            label_sums[label]['sum'] += dsc
            label_sums[label]['count'] += 1

    # Calculate and print the mean DSC for each label
    print("Mean DSC for each label:")
    for label in sorted(label_sums.keys()):
        mean_dsc = label_sums[label]['sum'] / label_sums[label]['count']
        print(f"Label {label}: {mean_dsc:.4f}")

    # Calculate overall mean DSC
    total_sum = sum(data['sum'] for data in label_sums.values())
    total_count = sum(data['count'] for data in label_sums.values())
    overall_mean = total_sum / total_count

    print(f"Overall Mean DSC: {overall_mean:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate mean DSC from CSV file.')
    parser.add_argument('--csv_path', type=str, help='Path to the CSV file')
    args = parser.parse_args()

    calculate_mean_dsc(args.csv_path)
