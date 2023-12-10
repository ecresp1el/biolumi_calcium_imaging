import os
import pandas as pd

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


    # Additional methods can be added here
