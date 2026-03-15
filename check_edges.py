import torch
import numpy as np
from pathlib import Path

def analyze_droid_graph(pth_path, window_threshold=50):
    # Load the reconstruction file
    data = torch.load(pth_path)
    
    # DROID-SLAM saves edges as (ii, jj) 
    # ii: indices of the first frames
    # jj: indices of the second frames
    if 'edges' not in data:
        print("Error: 'edges' not found in the .pth file. Ensure this is a full reconstruction save.")
        return

    ii = data['edges'][:, 0]
    jj = data['edges'][:, 1]

    # Calculate the 'distance' in time (frame index) between connected frames
    delta = np.abs(ii - jj)

    # Filter for loop closures: edges that span further than the local tracking window
    loop_mask = delta > window_threshold
    num_loop_edges = np.sum(loop_mask)
    total_edges = len(ii)

    # Calculate how many unique frames are involved in loop closures
    if num_loop_edges > 0:
        unique_frames_in_loops = len(np.unique(np.concatenate([ii[loop_mask], jj[loop_mask]])))
    else:
        unique_frames_in_loops = 0

    loop_edge_density = (num_loop_edges / total_edges) * 100 if total_edges > 0 else 0.0

    print(f"--- DROID-SLAM Graph Analysis ---")
    print(f"Traj:                  {pth_path.parent.name}")
    print(f"Total Edges in Graph:  {total_edges}")
    print(f"Loop Closure Edges:    {num_loop_edges}  (Threshold > {window_threshold})")
    print(f"Frames with Loops:     {unique_frames_in_loops}")
    print(f"Loop Edge Density:     {loop_edge_density:.2f}%")
    print(f"---------------------------------")

if __name__ == "__main__":
    # Replace with your actual path
    PATH = Path("/home/mbo/output_reconstruction/droidslam/droidslam-red/2024-11-21/red_2024-11-21-10-34/reconstruction.pth")
    analyze_droid_graph(PATH)