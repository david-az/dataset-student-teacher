import os
import pandas as pd
from azdet.utils.convert_annot import json_to_csv

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
    

def seen_during_training(frequencies, imgs_per_center, fictitious_len, n_epochs):
    """
    Prints the number of times each image is seen during training. 

    Args:
        frequencies (dict): A dictionary containing the frequency of images.
        imgs_per_center (dict): A dictionary containing the number of images for each center.
        fictitious_len (int): The number of images seen in one epoch.
        n_epochs (int): The number of epochs.

    """
    repeated_times = {}
    for center, current_freq in frequencies.items():
        n_imgs = imgs_per_center[center]
        imgs_seen_epoch = current_freq / sum(frequencies.values()) * fictitious_len
        imgs_repeated_times = round(n_epochs * imgs_seen_epoch / n_imgs)
        repeated_times[center] = imgs_repeated_times
    return repeated_times

def balance_frequencies(frequencies, imgs_per_center, fictitious_len, n_epochs, max_seen_epoch, min_seen_epoch, max_iteration=5):
    """
    Balances the frequency of images seen during training epochs. 

    Args:
        frequencies (dict): A dictionary containing the current frequency of images.
        imgs_per_center (dict): A dictionary containing the number of images for each center.
        fictitious_len (int): The number of images seen in one epoch.
        n_epochs (int): The number of epochs.
        max_seen_epoch (float): The maximum number of times an image can be seen in an epoch.
        min_seen_epoch (float): The minimum number of times an image can be seen in an epoch.
        max_iteration (int, optional): The maximum number of iterations for the balancing process. Defaults to 5.

    Returns:
        dict: A dictionary with the updated frequencies after balancing.
    """
    
    
    frequencies = frequencies.copy()
        
    for i in range(max_iteration):

        made_changes = False

        for center, current_freq in frequencies.items():
            
            n_imgs = imgs_per_center[center]
            imgs_seen_epoch = current_freq / sum(frequencies.values()) * fictitious_len

            imgs_repeated_times = round(n_epochs * imgs_seen_epoch / n_imgs)

            if imgs_repeated_times > max_seen_epoch * n_epochs:

                made_changes = True

                coef = (max_seen_epoch * n_imgs) / fictitious_len        
                sum_frequencies = sum(frequencies.values()) - frequencies[center]
                freq = (sum_frequencies * coef) / (1 - coef)
                frequencies[center] = freq

            elif imgs_repeated_times < min_seen_epoch * n_epochs:

                made_changes = True

                coef = (min_seen_epoch * n_imgs) / fictitious_len
                sum_frequencies = sum(frequencies.values()) - frequencies[center]
                freq = (sum_frequencies * coef) / (1 - coef)
                frequencies[center] = freq

            current_freq = frequencies[center]
            imgs_seen_epoch = current_freq / sum(frequencies.values()) * fictitious_len
            imgs_repeated_times = round(n_epochs * imgs_seen_epoch / n_imgs)

        if not made_changes:
            return frequencies

    return frequencies


def update_freq_relative_to_freq(freq_to_update, percent, main_freq):
    """Balance a dict of freq to represent a defined percentage when combined with another dict of freq
    
    Args:
        freq_to_update (dict): dict in the format:
            {
                "path/to/json": 1,
                "path/to/json": 3,
                ...
                "path/to/json": 2

            }

        percent (int): percentage that freq_to_update will be relative to main_freq, when both are combined
        main_freq (dict): dict in the same format as freq_to_update

    Returns:
        merged_freq (dict): merged dict of freq_to_update and main_freq
    
    """
    
    
    total_freq = sum(main_freq.values()) / (1 - percent) - sum(main_freq.values())
    coef_update = total_freq / sum(freq_to_update.values())
    freq_weighted = {path: freq * coef_update for path, freq in freq_to_update.items()}
    
    return dict(**main_freq, **freq_weighted)

def is_freq_dict(_dict):
    """Checks if a dict is a freq_dict or a config_dict
    """
    
    if isinstance(next(iter(_dict)), tuple):
        return False
        
    return True
        
def apply_global_freq(dict_freqs, depth=0):
    """Recursively merge dicts of freqs according to a config that specifies the percentage between
    the dicts of freqs
    
    Args:
        dict_freqs (dict): config in the following format. Each key (and subkey) of the config dict is a tuple
            that has two parts: the percentage and the name (the name is just for info) 
        
            dict_freqs = {
                (40, "pos"): {
                    (10, "frac_aug"): aug_frequencies,
                    (90, "frac_original"): {
                        (75, "center_a"): center_a_freq
                        (25, "center_b"): center_b_freq
                    }      
                },
                (60, "neg"): neg_frequencies
            }
            
        depth (int): not be used, simply for printing
        
    Returns:
        merged_freq (dict): merged dict of all frequencies in config

    """

    freqs_to_merge = []
    
    for percent, sub_items in dict_freqs.items():
        if is_freq_dict(sub_items):
            freqs_to_merge.append(sub_items)
        else:
            freqs_to_merge.append(apply_global_freq(sub_items, depth + 1))

    keys = list(dict_freqs.keys())
    assert len(keys) <= 2, 'Each key should have max two children' 
    percent1, name1 = keys[0]
    percent2, name2 = keys[1]

    new_merged_freq = update_freq_relative_to_freq(freqs_to_merge[0], percent1/100, freqs_to_merge[1])

    print(" " * 2 * depth, f"merged {name1} ({percent1}%) WITH {name2} ({percent2}%)")
    return new_merged_freq