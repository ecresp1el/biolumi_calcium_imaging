import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2

class ImageAnalysis:
    def __init__(self, project_folder):
        self.project_folder = project_folder
        self.directory_df = self.initialize_directory_df() 
        
    def initialize_directory_df(self):
        directories = [d for d in os.listdir(self.project_folder) if os.path.isdir(os.path.join(self.project_folder, d))]
        directory_data = [{'directory_name': d, 'directory_path': os.path.join(self.project_folder, d)} for d in directories]
        return pd.DataFrame(directory_data, columns=['directory_name', 'directory_path'])
    
    def list_directories(self):
        return [d for d in os.listdir(self.project_folder) if os.path.isdir(os.path.join(self.project_folder, d))]
    
    def list_files(self, folder_name):
        folder_path = os.path.join(self.project_folder, folder_name)
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                all_files.append(os.path.join(root, file))
        return all_files
    
    
    def generate_dark_image(self, tiff_path, num_frames=200):
        """
        Generates a median 'dark' image from the first specified number of frames in a multi-frame TIFF file.

        This method is used for compensating the dark pixel offset in bioluminescence imaging data.

        Parameters:
        tiff_path (str): Path to the multi-frame TIFF file.
        num_frames (int, optional): Number of frames to consider for generating the dark image. Defaults to 200.

        Returns:
        numpy.ndarray: A median image representing the 'dark' image.
        """
        with Image.open(tiff_path) as img:
            frames = [np.array(img.getdata(), dtype=np.float32).reshape(img.size[::-1]) for i in range(num_frames)]
            median_frame = np.median(frames, axis=0)
            return median_frame

    def subtract_dark_image(self, raw_tiff_path, dark_image):
        """
        Subtracts a 'dark' image from each frame of a multi-frame TIFF file.

        This method is used to compensate for the average dark pixel offset in bioluminescence imaging data.

        Parameters:
        raw_tiff_path (str): Path to the raw multi-frame TIFF file.
        dark_image (numpy.ndarray): The 'dark' image to be subtracted from each frame of the raw image.

        Returns:
        list of numpy.ndarray: A list of images, each representing a frame from the raw image with the dark image subtracted.
        """
        with Image.open(raw_tiff_path) as img:
            compensated_images = []
            for i in range(img.n_frames):
                img.seek(i)
                frame = np.array(img.getdata(), dtype=np.float32).reshape(img.size[::-1])
                compensated_image = cv2.subtract(frame, dark_image)
                compensated_images.append(compensated_image)
            return compensated_images

    # Additional methods can be added here


def print_cells_per_recording(sensor_data_map):
    """Print the number of cells analysed per recording for each sensor type."""
    for sensor_name, all_data in sensor_data_map.items():
        print(f"Sensor: {sensor_name}")

        if not all_data:
            print("  No recordings found.\n")
            continue

        total_cells = 0
        for session_id, session_data in sorted(all_data.items()):
            roi_data = session_data.get('roi_data')

            if isinstance(roi_data, dict):
                cell_count = len(roi_data)
            elif hasattr(roi_data, 'columns'):
                # Accept DataFrame-like objects that store ROI columns
                cell_count = sum(col.startswith('ROI') for col in roi_data.columns)
            else:
                cell_count = 0

            total_cells += cell_count
            print(f"  Recording {session_id}: {cell_count} cells")

        print(f"  Total cells: {total_cells}\n")
