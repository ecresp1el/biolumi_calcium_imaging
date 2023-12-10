import os

class ImageAnalysis:
    def __init__(self, project_folder):
        self.project_folder = project_folder

    def list_files(self, folder_name):
        folder_path = os.path.join(self.project_folder, folder_name)
        return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Additional methods can be added here
