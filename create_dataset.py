import numpy as np
import pandas as pd
import os
from azdet.utils.convert_annot import json_to_csv
from v9_frequencies import labeled_frequencies
from azdet.utils.convert_annot import csv_to_coco
import multiprocessing as mp

def get_imgs_per_path(dict_or_list, create_csv=True):
    """Helper function that returns a dict containing the number of images for each path
    """
    
    imgs_per_path = dict()

    for path in dict_or_list:
        csv_path = path.replace('json', 'csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, header=None)
        elif create_csv:
            json_to_csv(path, csv_path)
            df = pd.read_csv(csv_path, header=None)
        else:
            raise Exception(f"{csv_path} doesn't exist")

        imgs_per_path[path] = len(set(df[0]))

    return imgs_per_path

def split_pos_neg(input_dict):
    pos_dict = {}
    neg_dict = {}
    
    for key, value in input_dict.items():
        if key.split('/')[-1].startswith('train_'):
            pos_dict[key] = value
        elif key.split('/')[-1].startswith('removed_neg'):
            neg_dict[key] = value
    
    return pos_dict, neg_dict


def reduce_images_respecting_distribution(count_per_path, max_len):
    total_count = sum(count_per_path.values())
    
    if total_count <= max_len:
        return count_per_path  # No need to reduce if total count is already below max_len
    
    reduction_factor = max_len / total_count
    
    # First pass: reduce counts proportionally
    reduced_counts = {path: int(count * reduction_factor) for path, count in count_per_path.items()}
    
    # Second pass: distribute remaining slots
    remaining_slots = max_len - sum(reduced_counts.values())
    sorted_paths = sorted(count_per_path.items(), key=lambda x: x[1], reverse=True)
    
    for path, _ in sorted_paths:
        if remaining_slots <= 0:
            break
        if reduced_counts[path] < count_per_path[path]:
            reduced_counts[path] += 1
            remaining_slots -= 1
    
    # Remove entries with zero counts
    final_counts = {path: count for path, count in reduced_counts.items() if count > 0}
    
    return final_counts


count_per_path = get_imgs_per_path(labeled_frequencies)
pos_counts, neg_counts = split_pos_neg(count_per_path)

print(f"Positive samples: {sum(pos_counts.values())}")
print(f"Negative samples: {sum(neg_counts.values())}")

max_len = 50000  # Set your desired maximum length here

# only take files with at least 1000 images
pos_above_x = {k: v for k, v in pos_counts.items() if v > 1000}
limited_counts = reduce_images_respecting_distribution(pos_above_x, max_len)

# Print the results
print(f"Original total: {sum(pos_above_x.values())}")
print(f"Limited total: {sum(limited_counts.values())}")
print(f"Number of datasets: {len(pos_above_x)} -> {len(limited_counts)}")
print(f"Max value per dataset: {max(limited_counts.values())}")
print(f"Min value per dataset: {min(limited_counts.values())}")

pos_centers = [k.split('/')[-3] for k in limited_counts]

# only take files with at least 1000 images for negative samples and only those with pos annots as well
neg_above_x = {k: v for k, v in neg_counts.items() if v > 1000 and k.split('/')[-3] in pos_centers}
limited_neg_counts = reduce_images_respecting_distribution(neg_above_x, 150000)

# Print the results for negative samples
print("\nNegative samples:")
print(f"Original total: {sum(neg_above_x.values())}")
print(f"Limited total: {sum(limited_neg_counts.values())}")
print(f"Number of datasets: {len(neg_above_x)} -> {len(limited_neg_counts)}")
print(f"Max value per dataset: {max(limited_neg_counts.values())}")
print(f"Min value per dataset: {min(limited_neg_counts.values())}")

for counts in [limited_counts, limited_neg_counts]:
    for path, n in counts.items():
        csv_path = path.replace('json', 'csv')
        df = pd.read_csv(csv_path, header=None)
        selected_paths = df[0].sample(n=n, replace=False).tolist()
        df_trunc = df[df[0].isin(selected_paths)]
        output_path = os.path.join('data', os.path.basename(csv_path).replace('train_', 'train_reduced_').replace('removed_neg_', 'removed_neg_reduced_'))
        df_trunc.to_csv(output_path, index=False, header=False)


def process_file(csv_file):
    json_file = csv_file.replace('.csv', '.json')
    if not os.path.exists(json_file):
        csv_to_coco(csv_file, json_file)

# Get all CSV files in the data directory
csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]

# Create a pool of workers
pool = mp.Pool(processes=mp.cpu_count())

# Process files in parallel
pool.map(process_file, [os.path.join('data', f) for f in csv_files])

# Close the pool
pool.close()
pool.join()

print("Conversion of CSV files to COCO JSON format completed.")
