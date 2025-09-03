"""
This script processes ADST ROOT files to extract and transform cosmic ray shower data.

It performs the following steps:
1.  Loads event and station data from a ROOT file.
2.  Filters stations based on UUB presence and rejection status.
3.  Filters events to include only 6T5 events.
4.  Adds station position information.
5.  Sorts stations by total signal.
6.  Calculates station positions relative to the station with the highest signal.
7.  Converts Cartesian coordinates to axial coordinates.
8.  Maps axial coordinates to a 2D matrix.
9.  Sets saturated signals to zero.
10. Converts the primary particle ID to a binary classification (proton vs. iron).
11. Creates a "shower plane" matrix representing the detector signals.
12. Saves the processed data to two parquet files: one with detailed information
    and another with only the data needed for network training.
"""
from pathlib import Path
import uproot
import argparse
import numpy as np
import awkward as ak
import polars as pl

from tqdm import tqdm

import ROOT
ROOT.gROOT.SetBatch(True)

# The path to the library is depends on your computer, so the code WILL NOT RUN if you do not change this path.
LIB_PATH = "/home/joao/auger/software/Install/offline/4.0.1-icrc2023-prod1/lib/libRecEventKG.so"

def check_for_exps_folders(directory: Path, base: str) -> str:
    """
    Returns a unique directory name in the specified directory.

    Args:
        directory (str): The path to the folder where the new directory will
            be created.
        base (str): The base name for the new directory (e.g., "run").

    Returns:
        str: A unique directory name in the format "baseXX", where XX is a
            two-digit number.
    """
    counter = 1

    while True:
        candidate = f"{base}_{counter:02d}"
        if not (directory / candidate).exists():
            return candidate
        counter += 1

def parse_arguments():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process ADST ROOT files.")
    parser.add_argument('num_underscore', type=int, help='Number of underscores in the model name')
    parser.add_argument('file_path', type=Path, help='Path to the ADST ROOT file')
    return parser.parse_args()

def load_data(file_path):
    """Loads event and station data from a ROOT file.

    Args:
        file_path (str): The path to the ROOT file.

    Returns:
        tuple: A tuple containing two dictionaries:
               - event_conditions: A dictionary of event-level data.
               - station_conditions: A dictionary of station-level data.
    """
    adst = uproot.open(file_path)
    
    # Here is where I defined the variables that each event has. If you need different variables check the adst keys. 
    
    event_conditions = {
        'xmax': adst["recData/event./event.fGenShower/event.fGenShower.fXmax"].array(),
        'primary': adst["recData/event./event.fGenShower/event.fGenShower.fIDPrim"].array(),
        'mc_energy': adst["recData/event./event.fGenShower/event.fGenShower.Shower/event.fGenShower.fEnergy"].array(),
        'mc_zenith': adst["recData/event./event.fGenShower/event.fGenShower.Shower/event.fGenShower.fAxisCoreCS"].array()['fZ'],
        'rec_energy': adst["recData/event./event.fSDEvent/event.fSDEvent.fSdRecShower.RecShower/event.fSDEvent.fSdRecShower.Shower/event.fSDEvent.fSdRecShower.fEnergy"].array(),
        'rec_zenith': adst["recData/event./event.fSDEvent/event.fSDEvent.fSdRecShower.RecShower/event.fSDEvent.fSdRecShower.Shower/event.fSDEvent.fSdRecShower.fAxisCoreCS"].array()['fZ'],
        'sd_event_id': adst["recData/event./event.fSDEvent/event.fSDEvent.fSdEventId"].array(),
    }
    
    # Here is where I defined the variables that each station on the event has. If you need different variables
    # check the adst keys.
    
    station_conditions = {
        'station_id': adst["recData/event./event.fSDEvent/event.fSDEvent.fStations/event.fSDEvent.fStations.fId"].array(),
        'has_uub': adst["recData/event./event.fSDEvent/event.fSDEvent.fStations/event.fSDEvent.fStations.fIsUUB"].array(),
        'rejection_status': adst["recData/event./event.fSDEvent/event.fSDEvent.fStations/event.fSDEvent.fStations.fRejectionStatus"].array(),
        'saturation_status': adst["recData/event./event.fSDEvent/event.fSDEvent.fStations/event.fSDEvent.fStations.fSaturationStatus"].array(),
        'total_signal': adst["recData/event./event.fSDEvent/event.fSDEvent.fStations/event.fSDEvent.fStations.fTotalSignal"].array(),
        'scintillator_signal': adst["recData/event./event.fSDEvent/event.fSDEvent.fStations/event.fSDEvent.fStations.fScintillator.fTotalSignal"].array(),
        'scintillator_saturation': adst["recData/event./event.fSDEvent/event.fSDEvent.fStations/event.fSDEvent.fStations.fScintillator.fSaturationStatus"].array(),
    }
    
    return event_conditions, station_conditions

def filter_station_data(station_conditions):
    """Filters station data based on UUB presence and rejection status.

    Args:
        station_conditions (dict): A dictionary of station-level data.

    Returns:
        dict: A dictionary of filtered station-level data.
    """

    # My research is focused on AugerPrime, so I need that the station have UUB. Rejected stations are also 
    # removed from the array. If you want to use different cuts you need to change this condition.
    
    condition = (station_conditions['has_uub']) & (station_conditions['rejection_status'] == 0)
    

    filtered_station_conditions = {}
    for key, array in tqdm(station_conditions.items()):
        
        if key == 'rejection_status' or key == 'has_uub':
            continue 
        
        # The cuts actually happen here, the array[i][condition[i]] is a way to select only the values that are  
        # true for both conditions set before.
        
        filtered_array = [array[i][condition[i]] for i in range(len(array))]
        filtered_station_conditions[key] = [ak.to_numpy(a) for a in filtered_array]
    
    return filtered_station_conditions

def convert_event_conditions_to_numpy(event_conditions):
    """Converts event conditions from awkward arrays to numpy arrays.

    Args:
        event_conditions (dict): A dictionary of event-level data.

    Returns:
        dict: A dictionary of event-level data as numpy arrays.
    """
    return {key: array.to_numpy() for key, array in event_conditions.items()}

def create_dataframe(event_conditions_np, filtered_station_conditions):
    """Creates a Polars DataFrame from event and station data.

    Args:
        event_conditions_np (dict): A dictionary of event-level data as numpy arrays.
        filtered_station_conditions (dict): A dictionary of filtered station-level data.

    Returns:
        polars.DataFrame: The created DataFrame.
    """

    # This concatenate the two dictionaries
    
    data = {**event_conditions_np, **filtered_station_conditions}
    
    df = pl.DataFrame(data)
    
    # Filter empty events
    
    df = df.filter(pl.col("total_signal").list.len() > 0)
    
    return df

def filter_events(df, lib_path, file_path):
    """Filters events to include only 6T5 events.

    Args:
        df (polars.DataFrame): The DataFrame of events.
        lib_path (str): The path to the ROOT library.
        file_path (str): The path to the ROOT file.

    Returns:
        tuple: A tuple containing:
               - polars.DataFrame: The filtered DataFrame.
               - ROOT.DetectorGeometry: The detector geometry.
    """

    # Load ROOT library
    
    ROOT.gSystem.Load(lib_path)
    root_file = ROOT.RecEventFile(file_path, False)
    root_event = ROOT.RecEvent()
    root_file.SetBuffers(root_event)
    sd_event = root_event.GetSDEvent()
    det = ROOT.DetectorGeometry()
    root_file.ReadDetectorGeometry(det)
    
    # Get list of event IDs
    
    event_ids = df.select("sd_event_id").unique().to_numpy()[:,0]
    
    # Here the third cut is applied, I chose to do this way and not the same way as the other conditions, because
    # the number code used for in the dictionary was not clear for me. So i opt for this way, which is slower but 
    # I am sure the cut is properly done.
    
    drop_list = []
    for event_id in tqdm(event_ids):
        root_file.SearchForSDEvent(int(event_id))
        if not sd_event.Is6T5():
            drop_list.append(event_id)
    
    # Drop events from DataFrame
    
    df = df.filter(~pl.col("sd_event_id").is_in(drop_list))
    
    return df, det

def add_station_positions(df, det):
    """Adds station position information to the DataFrame.

    Args:
        df (polars.DataFrame): The DataFrame of events.
        det (ROOT.DetectorGeometry): The detector geometry.

    Returns:
        polars.DataFrame: The DataFrame with station positions.
    """

    # Get unique station IDs
    
    unique_station_ids = df.select("station_id").unique().to_numpy()[:,0]
    
    # Create a mapping from station ID to position
    
    position_dict = {}
    for station_id in tqdm(unique_station_ids):
        pos = det.GetStationPosition(int(station_id))
        position_dict[station_id] = [pos.X(), pos.Y(), pos.Z()]
    
    # Map positions to DataFrame
    
    pos_df = pl.DataFrame({
        "station_id": list(position_dict.keys()),
        "x": [position_dict[s][0] for s in position_dict],
        "y": [position_dict[s][1] for s in position_dict],
        "z": [position_dict[s][2] for s in position_dict],
    })
    df = df.join(pos_df, on="station_id")
    
    return df

def sort_by_total_signal(df):
    """Sorts the DataFrame by total signal in descending order for each event.

    Args:
        df (polars.DataFrame): The DataFrame of events.

    Returns:
        polars.DataFrame: The sorted DataFrame.
    """

    # The reason why I sort by sd_event_id first is to ensure that for each event the stations are sorted by total_signal
    # the scintillator_signal sort is only in case the total_signal of two stations are equal on the same event
    
    df = df.sort(by = ["sd_event_id", "total_signal", "scintillator_signal"], descending = [False, True, True])

    return df

def add_relative_positions(df):
    """Calculates station positions relative to the station with the highest signal.

    Args:
        df (polars.DataFrame): The DataFrame of events.

    Returns:
        polars.DataFrame: The DataFrame with relative positions.
    """
    
    # The way a chose to create the shower plane was that the highest energy station at the center and all the 
    # other stations positions are relative to the center station, so the (x_diff, y_diff) are the relative 
    # position.
    
    df = df.with_columns(
        (pl.col("x") - pl.first("x").over("sd_event_id")).alias("x_diff"),
        (pl.col("y") - pl.first("y").over("sd_event_id")).alias("y_diff")
    )
    
    return df


def cartesian_to_axial(df, apothem):
    """Converts Cartesian coordinates to axial coordinates.

    Args:
        df (polars.DataFrame): The DataFrame of events.
        apothem (float): The apothem of the hexagonal grid.

    Returns:
        polars.DataFrame: The DataFrame with axial coordinates.
    """
    
    # I am considering that each station is the center of one hexagon, so the apothem is half the distance
    # from one station to its neighbor.
    hex_radius = 2 * apothem / np.sqrt(3)

    # Here I transform the cartesian coordinates to hexagonal axial coordinates, I chose axial coordinates because  
    # the representation on 2d matrices are more memory efficient and the CNN kernel is more efficient.
    # I also opted for this way of doing due to it being faster and more memory efficient than the pythonic way.
    
    df = df.with_columns([
        ((np.sqrt(3) / 3 * pl.col("x_diff") - 1/3 * pl.col("y_diff")) / hex_radius)
        .round().alias("q"),

        ((2/3 * pl.col("y_diff")) / hex_radius)
        .round().alias("r")
    ])

    return df

def axial_to_matricial(df, matrix_size):
    """Maps axial coordinates to a 2D matrix.

    Args:
        df (polars.DataFrame): The DataFrame of events.
        matrix_size (int): The size of the output matrix.

    Returns:
        polars.DataFrame: The DataFrame with matrix coordinates.
    """

    # Finally I create the matrix with the axial coordinates. The station with the highest signal is in the center 
    # of the shower plane. The other are all placed on the around the center satition.

    matrix_center = (matrix_size - 1) / 2

    df = df.with_columns([
        (pl.col("q") + matrix_center).alias("row").cast(int),
        (pl.col("r") + matrix_center).alias("col").cast(int)
    ])

    return df

def remove_saturated_signals(df):
    """Sets saturated signals to zero.

    Args:
        df (polars.DataFrame): The DataFrame of events.

    Returns:
        polars.DataFrame: The DataFrame with saturated signals removed.
    """
    
    # Here I check if the signal of any station is saturated. If the signal is saturated I just change it to zero.
    # I did this cut at the end so even if higher signal station is saturated it will be at the center of the shower plane.
    
    df = df.with_columns(
        total_signal = pl.when(pl.col("saturation_status") < 2).then(0).otherwise(pl.col("total_signal")),
        scintillator_signal = pl.when(pl.col("scintillator_saturation") < 2).then(0).otherwise(pl.col("scintillator_signal"))
    )
    
    return df

def binary_primary(df):
    """Converts the primary particle ID to a binary classification.

    Args:
        df (polars.DataFrame): The DataFrame of events.

    Returns:
        polars.DataFrame: The DataFrame with a binary 'primary' column.
    """
    # Since I want to predict if the primary is "heavy" or "light" I change the primary to a binary code
    # the dataset I'm using only has iron and proton.

    return  df.with_columns(
                pl.when(pl.col("primary") == 2212)     #proton
                .then(0)
                .when(pl.col("primary") == 1000026056) #iron
                .then(1)
                .alias("primary")
            )

def create_shower_plane(df, matrix_size):
    """Creates a "shower plane" matrix representing the detector signals.

    Args:
        df (polars.DataFrame): The DataFrame of events.
        matrix_size (int): The size of the output matrix.

    Returns:
        polars.DataFrame: A DataFrame containing the shower plane matrices.
    """
    
    def shower_plane(args):
        """Updates the given array in-place based on the provided row/col/value mappings."""
        array_column = np.zeros((2,matrix_size,matrix_size))
        total_signal = args["total_signal"][0].to_numpy()
        scintillator_signal = args["scintillator_signal"][0].to_numpy()
        row = args["row"][0].to_numpy()
        col = args["col"][0].to_numpy()
    
        valid_mask = (row < matrix_size) & (col < matrix_size)

        array_column[0, row[valid_mask], col[valid_mask]] = total_signal[valid_mask]  # Update first channel
        array_column[1, row[valid_mask], col[valid_mask]] = scintillator_signal[valid_mask]  # Update second channel
        
        return pl.DataFrame({"sd_event_id": args["sd_event_id"], "xmax": args["xmax"], "primary": args["primary"], 
                            "mc_energy": args["mc_energy"], "mc_zenith": args["mc_zenith"],
                            "rec_energy": args["rec_energy"],
                            "rec_zenith": args["rec_zenith"], "shower_plane": array_column[np.newaxis, ...]})
    
    df = df.group_by([
            "sd_event_id", "xmax", "primary", "mc_energy", "mc_zenith", "rec_energy", "rec_zenith"
        ]).agg([
                pl.col("total_signal"), 
                pl.col("scintillator_signal"), 
                pl.col("row"), 
                pl.col("col")
            ])
    
    data = df.group_by("sd_event_id").map_groups(
                    lambda groupdf: shower_plane(groupdf)
            )
    
    return data

def main():
    """The main function of the script."""

    args = parse_arguments()
    file_path = args.file_path
    parts_num = args.num_underscore + 4

    # The matrix_size is hardcoded so that a typo does not cause big problems. Since the matrix is going to be the training
    # input for the CNN, so it must be equal for all events.
    
    matrix_size = 13
    apothem = 750

    base_name = file_path.stem
    parts = base_name.split('_')
    name = parts[:parts_num] + [parts[-1]]
    name = "_".join(name)

    save_path = file_path.parent.parent / "iterim"
    name = check_for_exps_folders(save_path, name)

    print("Loading ROOT file")
    event_conditions, station_conditions = load_data(file_path)
    print("ROOT file loaded")

    print("Start file filtering")
    filtered_station_conditions = filter_station_data(station_conditions)
    print("File filtering done")

    print("Converting events to numpy arrays")
    event_conditions_np = convert_event_conditions_to_numpy(event_conditions)
    print("Convertion completed")

    print("Creating DataFrame")
    df = create_dataframe(event_conditions_np, filtered_station_conditions)
    print("DataFrame created")

    df = df.explode(df.columns[-5:])

    print("Removing the non 6T5 events")
    df, det = filter_events(df, LIB_PATH, str(file_path))
    print("Events removed")

    print("Adding station position")
    df = add_station_positions(df, det)
    print("Positions added")

    print("Sorting by total signal")
    df = sort_by_total_signal(df)
    print("Sorted")

    print("Adding relative positions")
    df = add_relative_positions(df)
    print("Relative positions added")

    print("Adding axial coordinates")
    df = cartesian_to_axial(df, apothem)
    print("Axial coordinates added")

    print("Adding the positions at the matrix")
    df = axial_to_matricial(df, matrix_size)
    print("Positions added")

    print("Removing saturated signals")
    df = remove_saturated_signals(df)
    print("Signals removed.")

    df = binary_primary(df)

    print("Creating shower plane matrix")
    data = create_shower_plane(df, matrix_size)
    print("Shower plane created")

    # I chose parquet because its suitable for long storage and bigger DataFrames, also has the faster read and lazy read.
    # I chose to create two files because the first one has more information and the second is the actual data that will
    # be used on the network training.

    df.write_parquet(save_path / f"{name}.parquet")
    data.write_parquet(save_path / f"{name}_data.parquet")

if __name__ == '__main__':
    main()
