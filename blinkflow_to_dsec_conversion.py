import os
import numpy as np
import shutil
import h5py
import imageio
import argparse

def create_identity_rectify_map(height, width):
    """Create an identity rectify map with dimensions (height, width)."""
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    rectify_map = np.stack((x_coords, y_coords), axis=-1).astype(np.float32)
    return rectify_map

def save_rectify_map(rectify_map, output_path):
    """Save the rectify map to an HDF5 file."""
    with h5py.File(output_path, 'w') as h5f:
        h5f.create_dataset('rectify_map', data=rectify_map)

def store_flow_as_png(flow_data, output_path):
    """Store flow data as a PNG file."""
    H, W, _ = flow_data.shape
    validity = flow_data[..., 2].astype(np.uint16)
    flow_dx = flow_data[..., 0]
    flow_dy = flow_data[..., 1]
    
    flow_dx_int16 = (flow_dx * 128 + 2**15).astype(np.int16)
    flow_dy_int16 = (flow_dy * 128 + 2**15).astype(np.int16)
    
    flow_combined = np.zeros((H, W, 3), dtype=np.uint16)
    flow_combined[..., 0] = flow_dx_int16.astype(np.uint16)
    flow_combined[..., 1] = flow_dy_int16.astype(np.uint16)
    flow_combined[..., 2] = validity
    
    imageio.imwrite(output_path, flow_combined, format='PNG-FI')

def convert_blinkflow_sequence(group_folder, seq_folder, path_to_folder_to_convert, path_to_save_to):
    """Convert a single BlinkFlow sequence to the required format and save it."""
    path_to_convert = os.path.join(path_to_folder_to_convert, group_folder, seq_folder)
    flow_dir = os.path.join(path_to_convert, 'flow')
    forward_flow_dir = os.path.join(flow_dir, 'forward')
    
    os.makedirs(forward_flow_dir, exist_ok=True)
    files_to_convert = os.listdir(os.path.join(path_to_convert, 'forward_flow'))

    for file_to_convert in files_to_convert:
        if not file_to_convert.endswith('.npy'):
            continue
        data = np.load(os.path.join(path_to_convert, 'forward_flow', file_to_convert), allow_pickle=True)
        store_flow_as_png(data, os.path.join(forward_flow_dir, file_to_convert.split('.')[0] + ".png"))
    
    with open(os.path.join(flow_dir, 'forward_timestamps.txt'), 'w') as f:
        f.write("# from_timestamp_us, to_timestamp_us\n")
        for i in range(10):
            f.write(f"{i*100_000}, {(i+1)*100_000}\n")

    height, width = 480, 640
    rectify_map = create_identity_rectify_map(height, width)
    events_left_dir = os.path.join(path_to_convert, 'events_left')
    os.makedirs(events_left_dir, exist_ok=True)
    save_rectify_map(rectify_map, os.path.join(events_left_dir, 'rectify_map.h5'))

    path_to_save_events = os.path.join(path_to_save_to, 'train_events', group_folder + seq_folder)
    path_to_save_optical_flow = os.path.join(path_to_save_to, 'train_optical_flow', group_folder + seq_folder)

    shutil.copytree(flow_dir, os.path.join(path_to_save_optical_flow, 'flow'))
    shutil.copytree(events_left_dir, os.path.join(path_to_save_events, 'events', 'left'))

def convert_blinkflow_group(group_folder, path_to_folder_to_convert, path_to_save_to):
    """Convert all sequences in a group folder."""
    group_path = os.path.join(path_to_folder_to_convert, group_folder)
    sequences = [seq for seq in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, seq))]
    
    for seq_folder in sequences:
        print(f"Processing sequence: {seq_folder}")
        convert_blinkflow_sequence(group_folder, seq_folder, path_to_folder_to_convert, path_to_save_to)

def main():
    parser = argparse.ArgumentParser(description="Convert BlinkFlow sequences.")
    parser.add_argument("group_folder", type=str, help="Group folder to process.")
    parser.add_argument("path_to_folder_to_convert", type=str, help="Path to the folder containing data to convert.")
    parser.add_argument("path_to_save_to", type=str, help="Path to save the converted data.")

    args = parser.parse_args()

    convert_blinkflow_group(args.group_folder, args.path_to_folder_to_convert, args.path_to_save_to)

if __name__ == "__main__":
    main()
