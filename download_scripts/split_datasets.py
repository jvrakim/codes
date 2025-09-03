import os
import torch
import argparse
import numpy as np
import polars as pl

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Split the all parquet dataframes into a single dataframe.")
    parser.add_argument('data_path', type=str, help='Path to the whole dataset.')
    parser.add_argument('output_path', type=str, help='Path where the split datasets will be saved.')
    return parser.parse_args()

def get_data(path):
    df = pl.read_parquet(
            path,
            columns=["sd_event_id", "xmax", "primary", "shower_plane"]
    )

    xmax_array = torch.from_numpy(df["xmax"].cast(pl.Float32).to_numpy(writable=True))
    primary_array = torch.from_numpy(df["primary"].cast(pl.Int64).to_numpy(writable=True))

    shower_plane_shape = df["shower_plane"].dtype.shape
    shower_plane_array = torch.from_numpy(df["shower_plane"].cast(pl.Array(pl.Float32, shower_plane_shape)).to_numpy(writable=True))

    data_dict = {
        "sd_event_id": df["sd_event_id"].to_numpy(),
        "xmax_array": xmax_array, 
        "shower_plane_array": shower_plane_array, 
        "primary_array": primary_array
    }
    
    return data_dict

def create_balanced_splits(data_dict, train_percentage=0.7, val_size=0.5, test_size=0.5, random_state=42):
    """
    Create balanced splits with samples per class based on percentage of total dataset.
    Training set will have (train_percentage * T / N) samples per class, where:
    - T = total number of data points
    - N = number of classes
    
    Args:
        data_dict: Dictionary containing the data
        train_percentage: Percentage of total dataset to use for balanced training (e.g., 0.7 = 70%)
        val_size: Proportion for validation set (from remaining data after train)
        test_size: Proportion for test set (from remaining data after train)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with train/val/test indices
    """
    assert abs(val_size + test_size - 1.0) < 1e-6, "Val and test proportions must sum to 1.0"
    assert 0 < train_percentage < 1, "train_percentage must be between 0 and 1"
    
    # Get unique classes and their indices
    primary_classes = data_dict["primary_array"].numpy()
    unique_classes = np.unique(primary_classes)
    T = len(primary_classes)  # Total data points
    N = len(unique_classes)   # Number of classes
    
    # Calculate target samples per class
    target_samples_per_class = int((train_percentage * T) / N)
    
    # Find class sizes
    class_counts = {}
    class_indices_dict = {}
    
    print("Dataset Statistics:")
    print(f"Total data points (T): {T}")
    print(f"Number of classes (N): {N}")
    print(f"Train percentage (p): {train_percentage*100:.1f}%")
    print(f"Target samples per class: {target_samples_per_class} = ({train_percentage} * {T}) / {N}")
    print("\nAnalyzing class distributions:")
    
    insufficient_classes = []
    
    for class_label in unique_classes:
        class_indices = np.where(primary_classes == class_label)[0]
        class_counts[class_label] = len(class_indices)
        class_indices_dict[class_label] = class_indices
        
        status = "✓" if len(class_indices) >= target_samples_per_class else "✗"
        print(f"Class {class_label}: {len(class_indices)} samples {status}")
        
        if len(class_indices) < target_samples_per_class:
            insufficient_classes.append(class_label)
    
    # Check if any class has insufficient samples
    if insufficient_classes:
        print("\n❌ ERROR: Insufficient samples for balanced training!")
        print(f"Target samples per class: {target_samples_per_class}")
        print("Classes with insufficient samples:")
        for class_label in insufficient_classes:
            available = class_counts[class_label]
            print(f"  Class {class_label}: Available={available}, Needed={target_samples_per_class}, Deficit={target_samples_per_class - available}")
        
        print("\nSolutions:")
        print(f"1. Reduce train_percentage (currently {train_percentage*100:.1f}%)")
        print("2. Collect more data for insufficient classes")
        print("3. Use a different balancing strategy")
        
        raise ValueError(f"Cannot create balanced training set: {len(insufficient_classes)} out of {N} classes have insufficient samples")
    else:
        actual_samples_per_class = target_samples_per_class
        print("\n✓ All classes have sufficient samples for balanced training")
    
    train_indices = []
    remaining_indices = []
    
    print(f"\nCreating balanced training set with {actual_samples_per_class} samples per class:")
    
    # Create balanced training set
    np.random.seed(random_state)
    for class_label in unique_classes:
        class_indices = class_indices_dict[class_label].copy()
        np.random.shuffle(class_indices)
        
        # Select samples for training (balanced)
        train_class_indices = class_indices[:actual_samples_per_class]
        remaining_class_indices = class_indices[actual_samples_per_class:]
        
        train_indices.extend(train_class_indices)
        remaining_indices.extend(remaining_class_indices)
        
        print(f"Class {class_label}: {len(train_class_indices)} for train, {len(remaining_class_indices)} remaining")
    
    # Convert remaining indices to array and shuffle
    remaining_indices = np.array(remaining_indices)
    np.random.shuffle(remaining_indices)
    
    # Split remaining data between validation and test
    n_remaining = len(remaining_indices)
    n_val = int(n_remaining * val_size)
    
    val_indices = remaining_indices[:n_val]
    test_indices = remaining_indices[n_val:]
    
    # Convert train indices to array and shuffle
    train_indices = np.array(train_indices)
    np.random.shuffle(train_indices)
    
    print("\nFinal split sizes:")
    print(f"Train: {len(train_indices)} ({len(train_indices)/T*100:.1f}% of total)")
    print(f"Val: {len(val_indices)} ({len(val_indices)/T*100:.1f}% of total)")
    print(f"Test: {len(test_indices)} ({len(test_indices)/T*100:.1f}% of total)")
    print(f"Total used: {len(train_indices) + len(val_indices) + len(test_indices)}/{T} ({(len(train_indices) + len(val_indices) + len(test_indices))/T*100:.1f}%)")
    
    # Verify no overlapping indices within the splitting function
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("❌ INTERNAL ERROR: Found overlapping indices in split creation!")
        print(f"Train-Val overlap: {len(train_val_overlap)} indices")
        print(f"Train-Test overlap: {len(train_test_overlap)} indices") 
        print(f"Val-Test overlap: {len(val_test_overlap)} indices")
        raise ValueError("Split creation produced overlapping indices!")
    else:
        print("✓ Split indices are independent")
    
    return {
        "train_indices": train_indices,
        "val_indices": val_indices, 
        "test_indices": test_indices
    }

def standardize_data(data, mean, std):
    return data.sub(mean).div(std)

def compute_mean_std(array):
    array_mean = torch.mean(array)
    array_std = torch.std(array)  
    
    # Avoid division by zero
    array_std = torch.where(array_std == 0, torch.ones_like(array_std), array_std)
    return array_mean, array_std

def apply_standardization(data_dict, split_indices, stats):
    """
    Apply standardization to all splits using training statistics.
    
    Args:
        data_dict: Original data dictionary
        split_indices: Dictionary with train/val/test indices
        stats: Dictionary with mean/std statistics from training data
    
    Returns:
        Dictionary with standardized train/val/test datasets
    """
    datasets = {}
    
    # Map the split names correctly
    split_mapping = {
        "train_indices": "train",
        "val_indices": "val", 
        "test_indices": "test"
    }
    
    for split_name, indices in split_indices.items():
        # Extract data for current split
        split_data = {
            "sd_event_id": data_dict["sd_event_id"][indices],
            "xmax_array": data_dict["xmax_array"][indices].clone(),
            "shower_plane_array": data_dict["shower_plane_array"][indices].clone(),
            "primary_array": data_dict["primary_array"][indices].clone()
        }
        
        # Apply standardization using training statistics
        split_data["shower_plane_array"][:, 0, ...] = standardize_data(
            split_data["shower_plane_array"][:, 0, ...], 
            stats["channel_1_mean"], 
            stats["channel_1_std"]
        )
        split_data["shower_plane_array"][:, 1, ...] = standardize_data(
            split_data["shower_plane_array"][:, 1, ...], 
            stats["channel_2_mean"], 
            stats["channel_2_std"]
        )
        split_data["xmax_array"] = standardize_data(
            split_data["xmax_array"], 
            stats["xmax_mean"], 
            stats["xmax_std"]
        )
        
        # Use the correct key name
        correct_key = split_mapping[split_name]
        datasets[correct_key] = split_data
    
    return datasets

if __name__ == '__main__':
    args = parse_arguments()
    data_path = args.data_path
    output_path = args.output_path

    print("Loading data...")
    # Get and prepare data
    data_dict = get_data(data_path)
    
    print(f"Total samples: {len(data_dict['primary_array'])}")
    
    # Create balanced splits
    split_indices = create_balanced_splits(
        data_dict, 
        train_percentage=0.8,  # Use 80% of total dataset for balanced training
        val_size=0.5,  # 50% of remaining data for validation
        test_size=0.5,  # 50% of remaining data for test
        random_state=42
    )
    
    print("\nComputing standardization statistics from training data...")
    # Compute statistics for standardization using ONLY training data
    train_indices = split_indices["train_indices"]
    train_shower_plane = data_dict["shower_plane_array"][train_indices]
    train_xmax = data_dict["xmax_array"][train_indices]
    
    channel_1_mean, channel_1_std = compute_mean_std(train_shower_plane[:, 0, ...])
    channel_2_mean, channel_2_std = compute_mean_std(train_shower_plane[:, 1, ...])
    xmax_mean, xmax_std = compute_mean_std(train_xmax)
    
    stats = {
        "channel_1_mean": channel_1_mean,
        "channel_1_std": channel_1_std,
        "channel_2_mean": channel_2_mean,
        "channel_2_std": channel_2_std,
        "xmax_mean": xmax_mean,
        "xmax_std": xmax_std
    }
    
    print("Applying standardization to all splits...")
    # Apply standardization to all splits
    datasets = apply_standardization(data_dict, split_indices, stats)
    
    # Verify no overlapping indices
    print("Verifying dataset independence...")
    train_set = set(datasets["train"]["sd_event_id"])
    val_set = set(datasets["val"]["sd_event_id"]) 
    test_set = set(datasets["test"]["sd_event_id"])
    
    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("❌ ERROR: Found overlapping indices!")
        print(f"Train-Val overlap: {len(train_val_overlap)} indices")
        print(f"Train-Test overlap: {len(train_test_overlap)} indices") 
        print(f"Val-Test overlap: {len(val_test_overlap)} indices")
        raise ValueError("Datasets are not independent!")
    else:
        print("✓ All datasets are independent (no overlapping indices)")
    
    # Save datasets
    print("Saving datasets...")
    train_path = os.path.join(output_path, 'train_data.pt')
    val_path = os.path.join(output_path, 'val_data.pt')
    test_path = os.path.join(output_path, 'test_data.pt')

    torch.save(datasets["train"], train_path)
    torch.save(datasets["val"], val_path)
    torch.save(datasets["test"], test_path)
    
    # Also save the standardization statistics for future use
    standardization_path = os.path.join(output_path, 'standardization_stats.pt')
    torch.save(stats, standardization_path)
    
    print(f"Done! Files saved to {output_path}:")
    print("- train_data.pt")
    print("- val_data.pt") 
    print("- test_data.pt")
    print("- standardization_stats.pt")
    
    # Print class distribution verification
    print("\nClass distribution verification:")
    for split_name, split_data in datasets.items():
        unique, counts = torch.unique(split_data["primary_array"], return_counts=True)
        print(f"{split_name.capitalize()}:")
        for class_id, count in zip(unique.tolist(), counts.tolist()):
            print(f"  Class {class_id}: {count} samples")