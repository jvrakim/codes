"""
This script combines multiple parquet files into a single dataset.

It scans a directory for parquet files matching a specific pattern, combines them
into a single Polars DataFrame, replaces the 'sd_event_id' column with a unique
row index, and saves the resulting dataset to a new parquet file.
"""

import polars as pl
import argparse
import os

def parse_arguments():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Unify all parquet dataframes into a single dataframe.")
    parser.add_argument('file_path', type=str, help='Path to the directory containing the parquet files.')
    parser.add_argument('output_name', type=str, help='Name of the output file.')
    return parser.parse_args()

def combine_parquet_files(input_dir='.', output_file='dataset.parquet', haronic_model = None):
    """Scans a directory for parquet files, combines them, and saves the result.

    Args:
        input_dir (str, optional): The directory to scan for parquet files. Defaults to '.'.
        output_file (str, optional): The name of the output file. Defaults to 'dataset.parquet'.
        haronic_model (str, optional): A string to filter the parquet files. Defaults to None.
    """
    source_glob = os.path.join(input_dir, haronic_model + '*_data.parquet')

    print(f"Scanning for parquet files with pattern: {source_glob}")

    try:
        # Build the lazy query plan
        combined_lazy_df = (
            pl.scan_parquet(source_glob)
            .drop("sd_event_id")
            .with_row_index(name="sd_event_id")
        )

        output_file = f"{haronic_model}_{output_file}"

        output_path = os.path.join(input_dir, output_file)
        print(f"Executing query and saving combined dataset to {output_path}")

        # This operate in eager mode due to the fact that streaming egine does not suport the
        # with_row_index operation.
        # Execute the plan with .collect() and then write the result from RAM to disk.
        combined_lazy_df.collect().write_parquet(output_path)

        print("Successfully saved combined dataset.")

    except pl.exceptions.NoDataError:
        print(f"No parquet files found matching the pattern '{source_glob}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    args = parse_arguments()
    combine_parquet_files(args.file_path, args.output_name)

