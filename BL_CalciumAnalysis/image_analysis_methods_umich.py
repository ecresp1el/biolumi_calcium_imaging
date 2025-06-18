

class ImageAnalysis:
    
    """
    Provides methods for organizing and analyzing image data from various directory structures within a project folder.
    This class initializes by reading the project folder, creating a DataFrame that lists each directory and its path,
    and allows for further expansion to include specific image analysis metadata.

    Parameters:
    - project_folder (str): The path to the project folder containing various image data directories.

    """
    
    def __init__(self, project_folder):
        self.project_folder = project_folder
        self.directory_df = self.initialize_directory_df() 
        
    def initialize_directory_df(self):
        directories = [d for d in os.listdir(self.project_folder) if os.path.isdir(os.path.join(self.project_folder, d))]
        directory_data = [{'directory_name': d, 'directory_path': os.path.join(self.project_folder, d)} for d in directories]
        return pd.DataFrame(directory_data, columns=['directory_name', 'directory_path'])
    
    def expand_directory_df(self):
        # Add new columns with default empty lists
        self.directory_df['sensor_type'] = ''
        self.directory_df['session_id'] = ''
        self.directory_df['stimulation_ids'] = [[] for _ in range(len(self.directory_df))]
        self.directory_df['stimulation_frame_number'] = [[] for _ in range(len(self.directory_df))]

        for index, row in self.directory_df.iterrows():
            folder_name = row['directory_name']
            folder_path = row['directory_path']
            
            # Parse folder name for sensor type and session id
            parts = folder_name.split('_')
            sensor_type = 'gcamp8' if parts[0].startswith('g') else 'cablam'
            session_id = parts[0][1:] + parts[1]  # Assuming the first part is always the experiment ID

            # Update DataFrame with sensor_type and session_id
            self.directory_df.at[index, 'sensor_type'] = sensor_type
            self.directory_df.at[index, 'session_id'] = session_id

            # Check for CSV file ending in 'biolumi' or 'fluor'
            csv_filename = [f for f in os.listdir(folder_path) if (f.endswith('biolumi.csv') or f.endswith('fluor.csv'))]
            if csv_filename:
                csv_file_path = os.path.join(folder_path, csv_filename[0])
                df_csv = pd.read_csv(csv_file_path, header=None)
                stimulation_ids = df_csv.iloc[1].dropna().tolist()
                stimulation_frame_number = df_csv.iloc[0].dropna().tolist()

                # Update DataFrame with stimulation information
                self.directory_df.at[index, 'stimulation_ids'] = stimulation_ids
                self.directory_df.at[index, 'stimulation_frame_number'] = stimulation_frame_number

        return self.directory_df
        
    def max_projection_mean_values(self, tif_path):
        """
        Generates a maximum intensity projection based on the mean values of a multi-frame TIF file
        and saves it to a new subdirectory 'processed_data/processed_image_analysis_output'
        with a '_max_projection' suffix in the file name.

        Parameters:
        tif_path (str): Path to the multi-frame TIF file.

        Returns:
        str: Path to the saved maximum intensity projection image.
        """

        with Image.open(tif_path) as img:
            # Initialize a summing array with the shape of the first frame and float type for mean calculation
            sum_image = np.zeros((img.height, img.width), dtype=np.float32)

            # Sum up all frames
            for i in range(img.n_frames):
                img.seek(i)
                sum_image += np.array(img, dtype=np.float32)

            # Compute the mean image by dividing the sum by the number of frames
            mean_image = sum_image / img.n_frames
        
        # Define the new directory path
        processed_dir = os.path.join(os.path.dirname(tif_path), 'processed_data', 'processed_image_analysis_output')
        
        # Create the directory if it does not exist
        os.makedirs(processed_dir, exist_ok=True)
        
        # Create a new file path for the max projection image with the '_max_projection' suffix
        # The filename is extracted from tif_path and appended with '_max_projection.tif'
        file_name = os.path.basename(tif_path)
        max_proj_image_path = os.path.join(processed_dir, file_name.replace('.tif', '_max_projection.tif'))
       
        # Save the max projection image to the new file path
        Image.fromarray(mean_image).save(max_proj_image_path)
        
        print(f"Max projection saved: {max_proj_image_path}")
        # Return the path to the saved image
        return max_proj_image_path
    
    def analyze_all_sessions(self, function_to_apply):
        """
        Iterates over all session IDs in the directory DataFrame and applies the given function to each,
        skipping sessions that have already been processed.

        Parameters:
        function_to_apply (callable): Function to be applied to each session. It should accept a session ID.

        Returns:
        dict: A dictionary with session_ids as keys and function return values as values.
        """
        results = {}
        for session_id in self.directory_df['session_id']:
            try:
                result = function_to_apply(session_id)
                results[session_id] = result
            except Exception as e:
                print(f"An error occurred while processing session {session_id}: {e}")
        return results
    
    def analyze_session_max_projection(self, session_id):
        """
        Wrapper function to apply max_projection_mean_values to a session's TIF file,
        skipping if the max projection file already exists.

        Parameters:
        session_id (str): The session ID for which the TIF file will be processed.

        Returns:
        str: Path to the processed max projection TIFF file or a message indicating it was skipped.
        """
        tif_path = self.get_session_raw_data(session_id)
        if isinstance(tif_path, str) and tif_path.endswith('.tif'):
            # Check if max projection file already exists
            directory_path = os.path.dirname(tif_path)
            processed_dir = os.path.join(directory_path, 'processed_data', 'processed_image_analysis_output')
            max_proj_filename = os.path.basename(tif_path).replace('.tif', '_max_projection.tif')
            max_proj_path = os.path.join(processed_dir, max_proj_filename)
            
            if os.path.exists(max_proj_path):
                print(f"Max projection file already exists for session {session_id}. Skipping.")
                return max_proj_path
            else:
                print(f"Generating max projection for session {session_id}...")
                return self.max_projection_mean_values(tif_path)
        else:
            return f"No valid TIF file found for session {session_id}"
        
    def get_session_raw_data(self, session_id):
        # Check if the session_id is in the 'session_id' column of the directory_df
        if session_id in self.directory_df['session_id'].tolist():
            # Find the directory path for the given session_id
            directory_path = self.directory_df[self.directory_df['session_id'] == session_id]['directory_path'].values[0]
            
            # Search for the .tif file within that directory
            for file_name in os.listdir(directory_path):
                if file_name.endswith('.tif'):
                    return os.path.join(directory_path, file_name)

            # If no .tif file is found in the directory
            return f"No .tif file found in the directory for session {session_id}."
        else:
            # If the session_id is not present in the DataFrame
            return f"Session ID {session_id} is not present in the directory DataFrame."
         
    def add_tiff_dimensions(self):
        """
        Analyzes the dimensions of TIF files in the directory DataFrame and adds this data as new columns.
        """
        # Ensure the DataFrame has the columns for dimensions
        if 'x_dim' not in self.directory_df.columns:
            self.directory_df['x_dim'] = None
            self.directory_df['y_dim'] = None
            self.directory_df['z_dim_frames'] = None

        # Iterate over each session_id and update the dimensions
        for index, row in self.directory_df.iterrows():
            tif_path = self.get_session_raw_data(row['session_id'])
            if isinstance(tif_path, str) and tif_path.endswith('.tif'):
                try:
                    with Image.open(tif_path) as img:
                        self.directory_df.at[index, 'x_dim'] = img.width
                        self.directory_df.at[index, 'y_dim'] = img.height
                        # For z-dimension, count the frames
                        img.seek(0)  # Ensure the pointer is at the beginning
                        frames = 0
                        while True:
                            try:
                                img.seek(img.tell() + 1)
                                frames += 1
                            except EOFError:
                                break
                        self.directory_df.at[index, 'z_dim_frames'] = frames
                except Exception as e:
                    print(f"Could not process TIF dimensions for session {row['session_id']}: {e}")