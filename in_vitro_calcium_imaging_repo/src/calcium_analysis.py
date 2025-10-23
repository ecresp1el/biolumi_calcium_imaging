"""CSV-based calcium imaging analysis utilities."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind

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
        self.directory_df['sensor_type'] = ''
        self.directory_df['session_id'] = ''
        self.directory_df['stimulation_ids'] = [[] for _ in range(len(self.directory_df))]
        self.directory_df['stimulation_frame_number'] = [[] for _ in range(len(self.directory_df))]
        for index, row in self.directory_df.iterrows():
            folder_name = row['directory_name']
            folder_path = row['directory_path']
            parts = folder_name.split('_')
            sensor_type = 'gcamp8' if parts[0].startswith('g') else 'cablam'
            session_id = parts[0][1:] + parts[1]
            self.directory_df.at[index, 'sensor_type'] = sensor_type
            self.directory_df.at[index, 'session_id'] = session_id
            csv_filename = [f for f in os.listdir(folder_path) if f.endswith('biolumi.csv') or f.endswith('fluor.csv')]
            if csv_filename:
                csv_file_path = os.path.join(folder_path, csv_filename[0])
                df_csv = pd.read_csv(csv_file_path, header=None)
                stimulation_ids = df_csv.iloc[1].dropna().tolist()
                stimulation_frame_number = df_csv.iloc[0].dropna().tolist()
                self.directory_df.at[index, 'stimulation_ids'] = stimulation_ids
                self.directory_df.at[index, 'stimulation_frame_number'] = stimulation_frame_number
        return self.directory_df

    def process_all_sessions(self, use_corrected_data=False):
        """
        This assumes the analyze_all_rois method has been previously run to generate the numpy files 
        and the corresponding images with and without labels for ROI per session.
        
        Process all sessions using either corrected or uncorrected calcium signal data.

        Parameters
        ----------
        use_corrected_data : bool, optional
            Flag indicating whether to use corrected calcium signals. Defaults to False, 
            indicating uncorrected data should be used.

        Returns
        -------
        dict
            A dictionary with processed data for all sessions, keyed by session ID.
        """
        all_data = {}
        for session_id in self.directory_df['session_id'].unique():
            stim_frame_numbers, roi_data, stimulation_ids = self.create_trial_locked_calcium_signals(session_id, use_corrected_data=use_corrected_data)
            if stim_frame_numbers and roi_data and stimulation_ids:
                all_data[session_id] = {'stim_frame_numbers': stim_frame_numbers, 'roi_data': roi_data, 'stimulation_ids': stimulation_ids}
        return all_data

    def create_trial_locked_calcium_signals(self, session_id, use_corrected_data=False):
        """
        Generate trial-locked calcium signal data for a given session ID, allowing
        the choice between corrected and uncorrected data.
        
        Parameters
        ----------
        session_id : str
            The session ID for which to generate trial-locked signals.
        use_corrected_data : bool, optional
            Whether to use corrected calcium signal data. The default is False, which uses uncorrected data.
        
        Returns
        -------
        tuple
            A tuple containing the stimulation frame numbers, ROI data, and stimulation IDs.
        """
        processed_dir = 'processed_data/processed_image_analysis_output'
        calcium_csv_suffix = '_corrected_calcium_signals.csv' if use_corrected_data else '_calcium_signals.csv'
        directory_entry = self.directory_df[self.directory_df['session_id'] == session_id]
        if directory_entry.empty:
            print(f'No directory entry found for session {session_id}')
            return (None, None, None)
        directory_path = directory_entry['directory_path'].values[0]
        csv_path = os.path.join(directory_path, processed_dir, f'{session_id}{calcium_csv_suffix}')
        if not os.path.exists(csv_path):
            print(f"Calcium signals file not found for session {session_id} using {('corrected' if use_corrected_data else 'uncorrected')} data")
            return (None, None, None)
        calcium_signals_df = pd.read_csv(csv_path)
        stim_frame_numbers = directory_entry['stimulation_frame_number'].values[0]
        stimulation_ids = directory_entry['stimulation_ids'].values[0]
        pre_stim_frames = 10
        post_stim_frames = 100
        roi_data = {roi: {} for roi in calcium_signals_df.columns if 'ROI' in roi}
        for stim_id, stim_frame in zip(stimulation_ids, stim_frame_numbers):
            start_idx = max(stim_frame - pre_stim_frames, 0)
            end_idx = min(stim_frame + post_stim_frames, len(calcium_signals_df))
            for roi in roi_data:
                trial = calcium_signals_df.loc[start_idx:end_idx, roi]
                roi_data[roi][stim_id, stim_frame] = trial.to_numpy()
        return (stim_frame_numbers, roi_data, stimulation_ids)

    def process_all_sessions_entire_recording(self, use_corrected_data=False):
        """
        Processes all sessions and stores calcium signal dataframes in a dictionary.

        Parameters
        ----------
        use_corrected_data : bool, optional
            Whether to use corrected calcium signal data. The default is False, which uses uncorrected data.

        Returns
        -------
        dict
            A dictionary where each key is a session ID and the value is the corresponding calcium_signals dataframe.
        """
        processed_dir = 'processed_data/processed_image_analysis_output'
        calcium_csv_suffix = '_corrected_calcium_signals.csv' if use_corrected_data else '_calcium_signals.csv'
        session_data = {}
        for session_id in self.directory_df['session_id'].unique():
            directory_entry = self.directory_df[self.directory_df['session_id'] == session_id]
            if directory_entry.empty:
                print(f'No directory entry found for session {session_id}')
                continue
            directory_path = directory_entry['directory_path'].values[0]
            csv_path = os.path.join(directory_path, processed_dir, f'{session_id}{calcium_csv_suffix}')
            if not os.path.exists(csv_path):
                print(f"Calcium signals file not found for session {session_id} using {('corrected' if use_corrected_data else 'uncorrected')} data")
                continue
            calcium_signals_df = pd.read_csv(csv_path)
            session_data[session_id] = calcium_signals_df
        return session_data

    def calculate_responsiveness(self, all_data, pre_stim_frames=10, post_stim_frames=20, alpha=0.05, return_dataframe=False):
        """
        This function calculates and identifies responsive cells within calcium imaging data, applying statistical 
        tests to determine whether the change in signal post-stimulation is significant compared to the pre-stimulation 
        baseline. It stores detailed metrics including means, standard deviations, and p-values for each ROI across all sessions.

        Parameters:
        - all_data (dict): Nested dictionary containing the processed calcium signal data for multiple sessions, 
        structured with session IDs as top-level keys.
        - pre_stim_frames (int): The number of frames before the stimulus used to calculate the baseline signal.
        - post_stim_frames (int): The number of frames after the stimulus used for post-stimulus signal analysis.
        - alpha (float): The significance level used to determine if a response is statistically significant.
        - return_dataframe (bool): If set to True, the function also returns a pandas DataFrame containing the computed metrics.

        Returns:
        - dict: A nested dictionary containing calculated metrics for each session ID, ROI, and stimulus event. If 
        `return_dataframe` is True, it also returns a DataFrame alongside this dictionary.

        The output dictionary follows a multi-level structure:
        - Level 1 (Session Level): Keys are session IDs, and values are dictionaries containing data for each session.
        - Level 2 (ROI Level): Within each session dictionary, keys are ROIs, and values are dictionaries with metrics for each ROI.
        - Level 3 (Stimulus Event Level): For each ROI, keys are tuples of (stimulation_id, stim_frame_number), and values 
        are dictionaries containing the metrics calculated for each stimulus event.

        Metrics included for each stimulus event:
        - 'pre_stim_mean': Mean of the signal in the pre-stimulus period.
        - 'pre_stim_sd': Standard deviation of the signal in the pre-stimulus period.
        - 'post_stim_peak': Maximum signal value in the post-stimulus period (not normalized).
        - 'post_stim_sd': Standard deviation of the signal in the post-stimulus period, excluding the peak value.
        - 'p_value': P-value from the t-test comparing pre-stimulus and post-stimulus signals.
        - 'is_responsive': Boolean indicating whether the ROI is considered responsive based on the p-value being below alpha.

        
        Returns:
        dict or (dict, pd.DataFrame): A dictionary and optionally a DataFrame containing all metrics and SDs for each session ID, ROI, and stimulus.
        """
        responsiveness_data = {}
        dataframe_rows = []
        for session_id, session_data in all_data.items():
            session_responsiveness = {}
            for roi, roi_data in session_data['roi_data'].items():
                roi_responsiveness = {}
                for (stim_id, stim_frame), signal_data in roi_data.items():
                    if signal_data.size >= pre_stim_frames + post_stim_frames + 1:
                        pre_stim_signal = signal_data[:pre_stim_frames]
                        post_stim_signal = signal_data[pre_stim_frames + 1:pre_stim_frames + 1 + post_stim_frames]
                        delta_f_f_full_array = (signal_data - np.mean(signal_data[:pre_stim_frames])) / np.mean(signal_data[:pre_stim_frames])
                        pre_stim_mean = np.mean(pre_stim_signal)
                        pre_stim_sd = np.std(pre_stim_signal)
                        post_stim_peak = np.nanmax(post_stim_signal) if not np.isnan(np.nanmax(post_stim_signal)) else np.nan
                        post_stim_sd = np.std(post_stim_signal[1:])
                        post_stim_peak_index = np.nanargmax(post_stim_signal) if not np.isnan(post_stim_peak) else np.nan
                        post_stim_median = np.median(post_stim_signal)
                        pre_stim_median = np.median(pre_stim_signal)
                        post_stim_mean = np.mean(post_stim_signal)
                        delta_f_f_post_stim = (post_stim_signal - pre_stim_mean) / pre_stim_mean
                        peak_delta_f_f_post_stim = (post_stim_peak - pre_stim_mean) / pre_stim_mean
                        t_stat, p_value = ttest_ind(pre_stim_signal, post_stim_signal, equal_var=False)
                        is_responsive = p_value < alpha if not np.isnan(p_value) else False
                        half_peak_value = post_stim_peak / 2 if not np.isnan(post_stim_peak) else np.nan
                        half_rise_index = np.where(post_stim_signal >= half_peak_value)[0][0] if np.any(post_stim_signal >= half_peak_value) else np.nan
                        half_decay_index = np.where(post_stim_signal[post_stim_peak_index:] <= half_peak_value)[0][0] + post_stim_peak_index if post_stim_peak_index and np.any(post_stim_signal[post_stim_peak_index:] <= half_peak_value) else np.nan
                        time_to_peak = max(100, post_stim_peak_index * 100) if not np.isnan(post_stim_peak_index) else np.nan
                        half_rise_time = half_rise_index * 100 if not np.isnan(half_rise_index) else np.nan
                        half_decay_time = half_decay_index * 100 if not np.isnan(half_decay_index) else np.nan
                    roi_responsiveness[stim_id, stim_frame] = {'pre_stim_mean': pre_stim_mean, 'pre_stim_sd': pre_stim_sd, 'post_stim_peak': post_stim_peak, 'post_stim_sd': post_stim_sd, 'p_value': p_value, 'post_stim_mean': post_stim_mean, 'delta_f_f_post_stim': delta_f_f_post_stim, 'pre_stim_median': pre_stim_median, 'post_stim_median': post_stim_median, 'peak_delta_f_f_post_stim': peak_delta_f_f_post_stim, 'is_responsive': is_responsive}
                    dataframe_rows.append({'session_id': session_id, 'roi': roi, 'stimulation_id': stim_id, 'stim_frame_number': stim_frame, 'pre_stim_mean': pre_stim_mean, 'pre_stim_sd': pre_stim_sd, 'post_stim_peak': post_stim_peak, 'post_stim_sd': post_stim_sd, 'post_stim_mean': post_stim_mean, 'delta_f_f_post_stim': delta_f_f_post_stim, 'pre_stim_median': pre_stim_median, 'post_stim_median': post_stim_median, 'peak_delta_f_f_post_stim': peak_delta_f_f_post_stim, 'delta_f_f_full_array': delta_f_f_full_array, 'raw_signal': signal_data, 'p_value': p_value, 'time_to_peak': time_to_peak, 'half_rise_time': half_rise_time, 'half_decay_time': half_decay_time, 'is_responsive': is_responsive})
                session_responsiveness[roi] = roi_responsiveness
            responsiveness_data[session_id] = session_responsiveness
        if return_dataframe:
            responsiveness_df = pd.DataFrame(dataframe_rows)
            return (responsiveness_data, responsiveness_df)
        else:
            return responsiveness_data

    def filter_responsive_rois(self, all_data, responsiveness_data):
        """
        Creates a new data structure similar to all_data but excludes the data for non-responsive ROIs 
        for specific stimulation IDs, maintaining only responsive ROI data.

        Parameters:
        all_data (dict): Original dictionary with the complete dataset.
        responsiveness_data (dict): Dictionary containing responsiveness information for each ROI.

        Returns:
        dict: A new dictionary mirroring all_data's structure but excluding data for non-responsive ROIs per stimulus.
        """
        filtered_data = {}
        for session_id, session_content in all_data.items():
            filtered_data[session_id] = {'stim_frame_numbers': session_content['stim_frame_numbers'], 'roi_data': {}, 'stimulation_ids': session_content['stimulation_ids']}
            for roi, roi_data in session_content['roi_data'].items():
                filtered_roi_data = {}
                for stim_key, signal_data in roi_data.items():
                    if responsiveness_data[session_id][roi].get(stim_key, {}).get('is_responsive', False):
                        filtered_roi_data[stim_key] = signal_data
                if filtered_roi_data:
                    filtered_data[session_id]['roi_data'][roi] = filtered_roi_data
        return filtered_data

    def filter_responsive_rois_by_stimulation(self, session_data, responsiveness_df):
        filtered_data_by_session = {}
        responsive_df = responsiveness_df[responsiveness_df['is_responsive'] & (responsiveness_df['stimulation_id'] == 12)]
        grouped_responsive_df = responsive_df.groupby('session_id')
        for session_id, group in grouped_responsive_df:
            session_frames_list = []
            unique_rois = group['roi'].unique()
            session_df = session_data.get(session_id)
            if session_df is None:
                print(f'Session ID {session_id} not found in session_data.')
                continue
            for roi in unique_rois:
                roi_number = re.search('\\d+', roi)
                if not roi_number:
                    print(f'ROI format is incorrect for {roi}')
                    continue
                roi_column_name = f'ROI_{roi_number.group()}'
                if roi_column_name in session_df.columns:
                    roi_frames_df = session_df[[roi_column_name]].copy()
                    session_frames_list.append(roi_frames_df)
                else:
                    print(f'Column {roi_column_name} not found in session dataframe for session_id {session_id}.')
            if session_frames_list:
                combined_frames_df = pd.concat(session_frames_list, axis=1)
                filtered_data_by_session[session_id] = combined_frames_df
        return filtered_data_by_session

    def plot_session_time_series(self, filtered_data_by_session):
        for session_id, session_df in filtered_data_by_session.items():
            median_signal = session_df.median(axis=1, skipna=True)
            plt.figure(figsize=(10, 6))
            plt.title(f'Session ID: {session_id} Entire Recording ')
            plt.xlabel('Time (frames)')
            plt.ylabel('Signal (a.u.)')
            for column in session_df.columns:
                plt.plot(session_df.index, session_df[column], color='lightgrey', alpha=0.5, lw=0.5)
            plt.plot(session_df.index, median_signal, color='blue', label='Median Signal')
            plt.legend()
            plt.show()

    def process_biolumi_calcium_signal(self, session_id, directory_df):
        processed_dir = 'processed_data/processed_image_analysis_output'
        calcium_csv_suffix = '_calcium_signals.csv'
        directory_entry = directory_df[directory_df['session_id'] == session_id]
        if directory_entry.empty:
            print(f'No directory entry found for session {session_id}. Please check the session_id.')
            return None
        directory_path = directory_entry['directory_path'].values[0]
        csv_path = os.path.join(directory_path, processed_dir, str(session_id) + calcium_csv_suffix)
        if not os.path.exists(csv_path):
            print(f'Calcium signals file not found for session {session_id}')
            return None
        calcium_signals_df = pd.read_csv(csv_path)
        roi_columns = [roi for roi in calcium_signals_df.columns if 'ROI' in roi]
        grand_mean = calcium_signals_df[roi_columns].iloc[:300].mean().mean()
        for roi in roi_columns:
            calcium_signals_df[roi] = calcium_signals_df[roi] - grand_mean
            calcium_signals_df.loc[calcium_signals_df[roi] < 0, roi] = np.nan
        corrected_csv_path = os.path.join(directory_path, processed_dir, str(session_id) + '_corrected' + calcium_csv_suffix)
        calcium_signals_df.to_csv(corrected_csv_path, index=False)
        return calcium_signals_df

    def process_all_sessions_biolumi(self):
        unique_sessions = self.directory_df['session_id'].unique()
        for session_id in unique_sessions:
            print(f'Processing session ID: {session_id}')
            self.process_biolumi_calcium_signal(session_id, self.directory_df)
            print(f'Completed processing for session ID: {session_id}')

    def plot_stim_responsiveness(self, df, stim_ids=None, include='both', y_lim=None, x_lim=None, mean_color='black', figsize=(15, 5)):
        """
        Plots the delta F/F response for given stimulation IDs, filtering based on responsiveness if specified.
        Individual replicates are plotted in light grey, while the mean response is plotted in a user-defined color.
        Adds a red dotted line at the stimulation onset, considering the user-defined x-axis limits.
        Prints the number of responsive and unresponsive units for each stimulus ID on the plot.
        User can define the y-axis limits, x-axis limits, and the figure size.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the responsiveness data.
        - stim_ids (list): List of stimulation IDs to plot. If None, all unique IDs in the DataFrame will be used.
        - include (str): Can be 'responsive', 'non-responsive', or 'both' to filter units based on responsiveness.
        - y_lim (tuple): A tuple of (min, max) for y-axis limits. If None, limits are automatically determined.
        - x_lim (tuple): A tuple of (min, max) for x-axis limits. If None, defaults to the entire range of the data.
        - mean_color (str): Color for the mean response line.
        - figsize (tuple): Figure dimension as (width, height).

        Returns:
        - fig (plt.Figure): The created figure.
        """
        if stim_ids is None:
            stim_ids = sorted(df['stimulation_id'].unique())
        else:
            stim_ids = sorted(stim_ids)
        stim_index = 9
        total_frames = 111
        sampling_interval = 100
        n_stims = len(stim_ids)
        fig, axes = plt.subplots(1, n_stims, figsize=figsize, sharey=True)
        if n_stims == 1:
            axes = [axes]
        if y_lim:
            plt.setp(axes, ylim=y_lim)
        if x_lim is not None:
            new_x_lim = (x_lim[0] * sampling_interval, x_lim[1] * sampling_interval)
            plt.setp(axes, xlim=new_x_lim)
        for ax, stim_id in zip(axes, stim_ids):
            stim_df = df[df['stimulation_id'] == stim_id]
            if include != 'both':
                stim_df = stim_df[stim_df['is_responsive'] == (include == 'responsive')]
            delta_f_f_values = np.vstack(stim_df['delta_f_f_full_array'].values)
            time_vector = np.arange(-stim_index, total_frames - stim_index) * sampling_interval
            for trace in delta_f_f_values:
                ax.plot(time_vector, trace, color='lightgrey', linewidth=0.5)
            mean_response = np.nanmedian(delta_f_f_values, axis=0)
            ax.plot(time_vector, mean_response, color=mean_color, label=f'Stim ID {stim_id}')
            if x_lim is None or (0 >= x_lim[0] and 0 <= x_lim[1]):
                ax.axvline(x=0, color='red', linestyle='--', label='Stimulation Onset')
            num_responsive = len(stim_df[stim_df['is_responsive'] == True])
            num_unresponsive = len(stim_df[stim_df['is_responsive'] == False])
            info_text = f'Responsive: {num_responsive}'
            ax.text(0.95, 0.95, info_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round'))
            ax.set_title(f'Stim ID {stim_id}')
            ax.set_xlabel('ms', fontsize=24)
            ax.set_ylabel('ΔF/F$_o$', fontsize=24)
            ax.tick_params(axis='y', labelsize=18)
            ax.tick_params(axis='x', labelsize=18)
            ax.legend().remove()
        plt.tight_layout()

    def process_all_sessions_entire_recording_gcampbackgroundcorrected(self, sessions, rois_list, use_corrected_data=False):
        """
        Processes specified sessions, performs background correction using specified ROIs, and overwrites the original CSV file
        with the corrected data.

        Parameters
        ----------
        sessions : list
            A list of session IDs, e.g., ['2112232023'].
        rois_list : list of lists
            A list where each element is a list of ROIs to average and subtract for the corresponding session,
            e.g., [['ROI_11', 'ROI_12', 'ROI_13']].
        use_corrected_data : bool, optional
            Whether to use corrected calcium signal data. The default is False, which uses uncorrected data.

        Returns
        -------
        dict
            A dictionary where each key is a session ID and the value is the corresponding corrected calcium_signals dataframe.

        Notes
        -----
        After performing the background correction, this method will overwrite the original CSV file with the corrected data
        using the same file name and path. It replaces the file that was originally accessed.
        """
        processed_dir = 'processed_data/processed_image_analysis_output'
        calcium_csv_suffix = '_corrected_calcium_signals.csv' if use_corrected_data else '_calcium_signals.csv'
        session_data = {}
        for session_id, rois_to_average in zip(sessions, rois_list):
            if session_id not in self.directory_df['session_id'].unique():
                print(f'No directory entry found for session {session_id}')
                continue
            directory_entry = self.directory_df[self.directory_df['session_id'] == session_id]
            directory_path = directory_entry['directory_path'].values[0]
            csv_path = os.path.join(directory_path, processed_dir, f'{session_id}{calcium_csv_suffix}')
            if not os.path.exists(csv_path):
                print(f"Calcium signals file not found for session {session_id} using {('corrected' if use_corrected_data else 'uncorrected')} data")
                continue
            calcium_signals_df = pd.read_csv(csv_path)
            print(f'Session {session_id}: Number of columns before operation: {calcium_signals_df.shape[1]}')
            if not all((roi in calcium_signals_df.columns for roi in rois_to_average)):
                print(f'Some ROIs not found in session {session_id}: {rois_to_average}')
                continue
            rois_to_average_no_frame = [roi for roi in rois_to_average if roi != 'Frame']
            roi_mean = calcium_signals_df[rois_to_average_no_frame].mean(axis=1)
            columns_to_subtract = calcium_signals_df.columns.difference(['Frame'])
            calcium_signals_corrected = calcium_signals_df.copy()
            calcium_signals_corrected[columns_to_subtract] = calcium_signals_corrected[columns_to_subtract].subtract(roi_mean, axis=0)
            calcium_signals_corrected = calcium_signals_corrected.drop(columns=rois_to_average_no_frame)
            print(f'Session {session_id}: Number of columns after operation: {calcium_signals_corrected.shape[1]}')
            session_data[session_id] = calcium_signals_corrected
            calcium_signals_corrected.to_csv(csv_path, index=False)
            print(f'Session {session_id}: Corrected data saved and overwrote {csv_path}')
        return session_data

    def plot_session_time_series_for_specific_rois(self, filtered_data_by_session, rois_to_include, frame_range=None):
        """
        Plots the time series for selected ROIs in each session with distinct colors for each ROI, 
        and converts frame numbers into time (seconds) on the x-axis based on 10Hz recording frequency.
        :param filtered_data_by_session: A dictionary where keys are session IDs and values are DataFrames containing ROI time series data.
        :param rois_to_include: A list of ROI names that should be included in the plot.
        :param frame_range: A tuple (start_frame, end_frame) specifying the range of frames to plot. If None, plot all frames.
        """
        colors = plt.cm.get_cmap('tab10', len(rois_to_include))
        for session_id, session_df in filtered_data_by_session.items():
            session_df_filtered = session_df[session_df.columns.intersection(rois_to_include)]
            if session_df_filtered.empty:
                print(f'No matching ROIs found in session {session_id} for the provided list of ROIs.')
                continue
            if frame_range is not None:
                start_frame, end_frame = frame_range
                session_df_filtered = session_df_filtered.iloc[start_frame:end_frame]
            else:
                start_frame, end_frame = (0, len(session_df_filtered))
            time_vector = session_df_filtered.index / 10.0
            plt.figure(figsize=(10, 6))
            plt.title(f'Session ID: {session_id} - Selected ROIs')
            plt.xlabel(f'Time (seconds)')
            plt.ylabel('Signal (a.u.)')
            for idx, column in enumerate(session_df_filtered.columns):
                plt.plot(time_vector, session_df_filtered[column], color=colors(idx), label=f'{column}', lw=1.5)
            plt.legend(loc='best')
            plt.show()

    def compare_sessions_time_series(self, session_data_list, roi_lists, frame_ranges=None, session_labels=None, fig_size=(12, 6), dpi=300, save_dir=None, save_dpi=300, responsiveness_dfs=None):
        """
        Plots time series for selected ROIs from multiple sessions in a vertical arrangement of subplots,
        with independent x-axes for each subplot and indicators for stimulation times.
        
        :param session_data_list: List of dictionaries, each containing session data (output from filter_responsive_rois_by_stimulation)
        :param roi_lists: List of lists, each containing ROI names to plot for the corresponding session
        :param frame_ranges: List of tuples, each specifying (start_frame, end_frame) for the corresponding session. 
                            If None, all frames will be plotted for each session.
        :param session_labels: List of labels for each session. If None, default labels will be used.
        :param fig_size: Tuple representing the figure size (width, height) in inches.
        :param dpi: The resolution in dots per inch for display.
        :param save_dir: Directory to save the plot. If None, the plot is not saved.
        :param save_dpi: The resolution in dots per inch for saving the figure.
        :param responsiveness_dfs: List of DataFrames containing responsiveness data for each session.
        """
        num_sessions = len(session_data_list)
        if frame_ranges is None:
            frame_ranges = [None] * num_sessions
        if session_labels is None:
            session_labels = [f'Session {i + 1}' for i in range(num_sessions)]
        if responsiveness_dfs is None:
            responsiveness_dfs = [None] * num_sessions
        fig, axes = plt.subplots(num_sessions, 1, figsize=(fig_size[0], fig_size[1] * num_sessions), sharex=False, dpi=dpi)
        if num_sessions == 1:
            axes = [axes]
        for idx, (session_data, rois_to_include, frame_range, session_label, responsiveness_df) in enumerate(zip(session_data_list, roi_lists, frame_ranges, session_labels, responsiveness_dfs)):
            ax = axes[idx]
            if session_label not in session_data:
                print(f'Session {session_label} not found in the provided data.')
                continue
            session_df = session_data[session_label]
            session_df_filtered = session_df[session_df.columns.intersection(rois_to_include)]
            if session_df_filtered.empty:
                print(f'No matching ROIs found in {session_label} for the provided list of ROIs.')
                continue
            if frame_range is not None:
                start_frame, end_frame = frame_range
                session_df_filtered = session_df_filtered.iloc[start_frame:end_frame]
            else:
                start_frame, end_frame = (0, len(session_df_filtered))
            start_time = start_frame / 10.0
            end_time = end_frame / 10.0
            time_vector = np.arange(len(session_df_filtered)) / 10.0
            colors = plt.cm.get_cmap('tab10', len(session_df_filtered.columns))
            for roi_idx, column in enumerate(session_df_filtered.columns):
                ax.plot(time_vector, session_df_filtered[column], color=colors(roi_idx), label=f'{column}', lw=1.5)
            if responsiveness_df is not None:
                stim_data = responsiveness_df[(responsiveness_df['session_id'] == session_label) & responsiveness_df['roi'].isin(rois_to_include) & responsiveness_df['stim_frame_number'].between(start_frame, end_frame)][['stim_frame_number', 'stimulation_id']].drop_duplicates()
                for _, stim_row in stim_data.iterrows():
                    stim_frame = stim_row['stim_frame_number']
                    stim_id = stim_row['stimulation_id']
                    stim_time = (stim_frame - start_frame) / 10.0
                    ax.axvline(x=stim_time, color='red', linestyle=':', linewidth=0.5, alpha=0.7)
                    ax.text(stim_time, ax.get_ylim()[1], f'{stim_id}', rotation=90, va='bottom', ha='right', fontsize=6)
            ax.set_ylabel('Signal (a.u.)')
            ax.set_title(f'Session {session_label}')
            ax.legend(loc='upper right')
            ax.set_xlabel('Time (seconds)')
            ax.set_xlim(0, end_time - start_time)
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.rcParams['svg.fonttype'] = 'none'
            session_names = '_'.join(session_labels)
            filename = f'{save_dir}/compare_sessions_{session_names}.svg'
            plt.savefig(filename, format='svg', dpi=save_dpi, transparent=True, bbox_inches='tight')
            print(f'Figure saved as {filename}')
        plt.show()

    def plot_time_locked_responses(self, session_data_list, roi_lists, session_labels, stim_ids, fig_size=(20, 15), dpi=300, save_dir=None, save_dpi=300):
        """
        Plots time-locked evoked responses for specified ROIs and stimulation IDs within sessions in a grid layout,
        with y-axis scaling based only on the specified ROIs and stim IDs for each session.
        
        :param session_data_list: List of DataFrames containing session data
        :param roi_lists: List of lists, each containing ROI names to plot for the corresponding session
        :param session_labels: List of labels for each session
        :param stim_ids: List of stimulation IDs to plot
        :param fig_size: Tuple representing the figure size (width, height) in inches
        :param dpi: The resolution in dots per inch for display
        :param save_dir: Directory to save the plots. If None, the plots are not saved
        :param save_dpi: The resolution in dots per inch for saving the figures
        """
        if isinstance(stim_ids, int):
            stim_ids = [stim_ids]
        for session_idx, (session_data, rois, session_label) in enumerate(zip(session_data_list, roi_lists, session_labels)):
            filtered_data = session_data[session_data['roi'].isin(rois) & session_data['stimulation_id'].isin(stim_ids)]
            all_delta_f = np.concatenate(filtered_data['delta_f_f_full_array'].values)
            all_delta_f = all_delta_f[np.isfinite(all_delta_f)]
            if len(all_delta_f) == 0:
                print(f'Warning: No valid data found for session {session_label}')
                continue
            y_min, y_max = (np.min(all_delta_f), np.max(all_delta_f))
            y_range = y_max - y_min
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range
            n_rows = len(rois)
            n_cols = len(stim_ids)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, dpi=dpi, sharex=True, sharey=True)
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            if n_cols == 1:
                axes = axes.reshape(-1, 1)
            time_vector = (np.arange(111) - 9) / 10
            for row, roi in enumerate(rois):
                for col, stim_id in enumerate(stim_ids):
                    ax = axes[row, col]
                    roi_stim_data = filtered_data[(filtered_data['roi'] == roi) & (filtered_data['stimulation_id'] == stim_id)]
                    if roi_stim_data.empty:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                        continue
                    delta_f_array = np.array(roi_stim_data['delta_f_f_full_array'].iloc[0])
                    ax.plot(time_vector, delta_f_array, color='black')
                    ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8)
                    if col == 0:
                        ax.set_ylabel(f'ROI {roi}\nΔF/F₀', fontsize=8)
                    if row == n_rows - 1:
                        ax.set_xlabel('Time (s)', fontsize=8)
                    if row == 0:
                        ax.set_title(f'Stim {stim_id}', fontsize=10)
                    ax.tick_params(axis='both', which='major', labelsize=6)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_ylim(y_min, y_max)
            plt.tight_layout()
            fig.suptitle(f'Session {session_label} - Time-locked Responses', fontsize=16, y=1.02)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.rcParams['svg.fonttype'] = 'none'
                filename = f'{save_dir}/session_{session_label}_time_locked_grid.svg'
                plt.savefig(filename, format='svg', dpi=save_dpi, transparent=True, bbox_inches='tight')
                print(f'Figure saved as {filename}')
            plt.show()
        for session_data, session_label in zip(session_data_list, session_labels):
            found_stims = set(session_data['stimulation_id'])
            missing_stims = set(stim_ids) - found_stims
            if missing_stims:
                print(f'Warning: Stimulation IDs {missing_stims} not found in Session {session_label}')

class SensorDataPlotter:

    def __init__(self, data_frames, sensor_names, sensor_box_colors, sensor_strip_colors):
        """
        Initialize the object with a list of data frames, corresponding sensor names, and specific colors for each sensor.
        :param data_frames: List of pandas DataFrames containing the sensor data.
        :param sensor_names: List of strings representing the names of the sensors.
        :param sensor_box_colors: Dictionary mapping sensor names to boxplot colors.
        :param sensor_strip_colors: Dictionary mapping sensor names to stripplot colors.
        """
        self.data_frames = data_frames
        self.sensor_names = sensor_names
        self.sensor_box_colors = sensor_box_colors
        self.sensor_strip_colors = sensor_strip_colors
        self.combined_df = None
        self._add_peak_snr_column()

    def _add_peak_snr_column(self):
        """
        Add a new column 'peak_snr' to each DataFrame in self.data_frames.
        The peak SNR is calculated as peak_delta_f_f_post_stim divided by pre_stim_sd.
        """
        for df in self.data_frames:
            df['peak_snr'] = df['post_stim_peak'] / df['pre_stim_sd']

    def prepare_for_plotting(self, df_column_name):
        """
        Prepares a single dataframe suitable for plotting from multiple sensor dataframes.
        :param df_column_name: The name of the column to use for the value in the plot.
        """
        frames = []
        for df, name in zip(self.data_frames, self.sensor_names):
            df = df.copy()
            df['sensor_name'] = name
            df['value'] = df[df_column_name]
            frames.append(df)
        self.combined_df = pd.concat(frames, ignore_index=True)
        self.combined_df = self.combined_df[self.combined_df['is_responsive'] == True]
        if df_column_name not in self.combined_df.columns:
            raise ValueError(f"The column '{df_column_name}' does not exist in the DataFrame.")
        self.combined_df[df_column_name] = pd.to_numeric(self.combined_df[df_column_name], errors='coerce')

    def plot_data(self, df_column_name, selected_stim_ids, box_width=0.8, strip_size=3, fig_size=(12, 8), dpi=300, save_dir=None, save_dpi=300, y_range=None):
        """
        Plots the data using boxplot and stripplot for selected stimulation IDs.
        :param df_column_name: The name of the column to use for the value in the plot.
        :param selected_stim_ids: List of stimulation IDs to plot. If None, plot all.
        :param box_width: The width of the boxplots.
        :param strip_size: The size of the points in the stripplots.
        :param fig_size: Tuple representing the figure size (width, height) in inches.
        :param dpi: The resolution in dots per inch for display.
        :param save_dir: Directory to save the plot. If None, the plot is not saved.
        :param save_dpi: The resolution in dots per inch for saving the figure.
        :param y_range: Tuple representing the y-axis limits (min, max). If None, limits are automatically determined.
        """
        df_column_name = str(df_column_name)
        if selected_stim_ids is not None:
            if not isinstance(selected_stim_ids, list):
                raise ValueError('The selected_stim_ids parameter must be a list of stimulation IDs.')
            if not all((isinstance(stim_id, int) for stim_id in selected_stim_ids)):
                raise ValueError('All elements in the selected_stim_ids list must be integers.')
        if not isinstance(box_width, (int, float)):
            raise ValueError('The box_width parameter must be an integer or float.')
        if not isinstance(strip_size, (int, float)):
            raise ValueError('The strip_size parameter must be an integer or float.')
        if not isinstance(fig_size, tuple) or len(fig_size) != 2:
            raise ValueError('The fig_size parameter must be a tuple of two integers.')
        if not all((isinstance(val, (int, float)) for val in fig_size)):
            raise ValueError('The fig_size parameter must contain only integers or floats.')
        if not isinstance(dpi, int):
            raise ValueError('The dpi parameter must be an integer.')
        if self.combined_df is None:
            self.prepare_for_plotting(df_column_name)
        if selected_stim_ids is not None:
            self.combined_df = self.combined_df[self.combined_df['stimulation_id'].isin(selected_stim_ids)]
        if self.combined_df.empty:
            raise ValueError('No data available for the selected stimulation IDs.')
        print(f"Unique stimulation IDs in the combined dataset: {sorted(self.combined_df['stimulation_id'].unique())}")
        if selected_stim_ids is not None:
            present_stim_ids = set(selected_stim_ids).intersection(set(self.combined_df['stimulation_id']))
            missing_stim_ids = set(selected_stim_ids) - present_stim_ids
            print(f'Stimulation IDs present in the data: {sorted(present_stim_ids)}')
            print(f'Stimulation IDs not found in the data: {sorted(missing_stim_ids)}')
            self.combined_df = self.combined_df[self.combined_df['stimulation_id'].isin(selected_stim_ids)]
        boxprops = {'edgecolor': 'k', 'linewidth': 1.5}
        lineprops = {'color': 'k', 'linewidth': 1.5}
        boxplot_kwargs = {'boxprops': boxprops, 'medianprops': lineprops, 'whiskerprops': lineprops, 'capprops': lineprops, 'width': box_width, 'palette': self.sensor_box_colors, 'hue_order': self.sensor_names}
        stripplot_kwargs = {'linewidth': 0.1, 'size': strip_size, 'alpha': 0.3, 'palette': self.sensor_strip_colors, 'hue_order': self.sensor_names}
        plt.figure(figsize=fig_size, dpi=dpi)
        ax = plt.subplot()
        sns.stripplot(x='stimulation_id', y=df_column_name, hue='sensor_name', data=self.combined_df, ax=ax, jitter=0.3, dodge=True, **stripplot_kwargs)
        sns.boxplot(x='stimulation_id', y=df_column_name, hue='sensor_name', data=self.combined_df, ax=ax, fliersize=0, **boxplot_kwargs)
        ax.set_xlabel('Stimulation ID', fontsize=18)
        ax.set_ylabel(df_column_name, fontsize=18)
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        ax.legend_.remove()
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.tight_layout()
        if y_range is not None:
            ax.set_ylim(y_range)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.rcParams['svg.fonttype'] = 'none'
            sensor_names = '_'.join(self.sensor_names)
            stim_ids = '_'.join(map(str, selected_stim_ids))
            filename = f'{save_dir}/{sensor_names}_{df_column_name}_stim_{stim_ids}.svg'
            plt.savefig(filename, format='svg', dpi=save_dpi, transparent=True, bbox_inches='tight')
            print(f'Figure saved as {filename}')
        plt.show()

    def plot_time_series(self, full_array_column, selected_stim_ids=None, fig_size=(6.5, 8), dpi=300, y_limits=None, save_dir=None, save_dpi=300, plot_sem=False, plot_sem_as_dotted=False):
        if self.combined_df is None:
            raise ValueError('Data has not been prepared for plotting. Call prepare_for_plotting first.')
        stim_ids = selected_stim_ids if selected_stim_ids is not None else self.combined_df['stimulation_id'].unique()
        num_stim_ids = len(stim_ids)
        num_sensors = len(self.sensor_names)
        max_sample_size = max(self.combined_df[full_array_column].apply(lambda x: len(x)))
        frame_rate = 10
        time_vector = (np.arange(max_sample_size) - 10) / frame_rate
        fig, axes = plt.subplots(num_sensors, num_stim_ids, figsize=(fig_size[0] * num_stim_ids, fig_size[1] * num_sensors), dpi=dpi, sharey=True, sharex=True)
        if num_sensors == 1 and num_stim_ids == 1:
            axes = np.array([[axes]])
        elif num_sensors == 1:
            axes = np.array(axes).reshape(1, num_stim_ids)
        elif num_stim_ids == 1:
            axes = np.array(axes).reshape(num_sensors, 1)
        for row_idx, sensor_name in enumerate(self.sensor_names):
            for col_idx, stim_id in enumerate(stim_ids):
                ax = axes[row_idx, col_idx]
                sensor_stim_data = self.combined_df[(self.combined_df['sensor_name'] == sensor_name) & (self.combined_df['stimulation_id'] == stim_id)]
                if not sensor_stim_data.empty:
                    responses = []
                    for _, row in sensor_stim_data.iterrows():
                        data = row[full_array_column]
                        if len(data) < max_sample_size:
                            data = np.pad(data, (0, max_sample_size - len(data)), 'constant', constant_values=np.nan)
                        responses.append(data)
                    responses = np.array(responses)
                    if not plot_sem:
                        for response in responses:
                            ax.plot(time_vector, response, color='gainsboro', alpha=0.3)
                    median_response = np.nanmedian(responses, axis=0)
                    ax.plot(time_vector, median_response, color=self.sensor_box_colors[sensor_name], label='Median')
                    if plot_sem:
                        sem_response = np.nanstd(responses, axis=0) / np.sqrt(np.sum(~np.isnan(responses), axis=0))
                        if plot_sem_as_dotted:
                            ax.plot(time_vector, median_response + sem_response, linestyle='--', color=self.sensor_box_colors[sensor_name], linewidth=0.7)
                            ax.plot(time_vector, median_response - sem_response, linestyle='--', color=self.sensor_box_colors[sensor_name], linewidth=0.7)
                        else:
                            ax.fill_between(time_vector, median_response - sem_response, median_response + sem_response, color=self.sensor_box_colors[sensor_name], alpha=0.2, label='SEM')
                    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
                    stim_duration = stim_id / 1000.0
                    if stim_duration > 0:
                        ax.axvspan(0, stim_duration, color='grey', alpha=0.1, zorder=1)
                        ax.axvline(0, color='red', linestyle='--', linewidth=0.8, zorder=2)
                        if stim_duration != 0.012:
                            ax.axvline(stim_duration, color='red', linestyle='--', linewidth=0.8, zorder=2)
                    else:
                        ax.axvline(0, color='red', linestyle='--', linewidth=0.8, zorder=2)
                else:
                    ax.axis('off')
                ax.set_title(f'Stim {stim_id} ms - {sensor_name}', fontsize=14)
                ax.set_xlabel('Time (s)', fontsize=18)
                if col_idx == 0:
                    ax.set_ylabel('ΔF/F', fontsize=18)
                if y_limits is not None:
                    ax.set_ylim(y_limits)
                ax.tick_params(axis='x', labelsize=14)
                ax.tick_params(axis='y', labelsize=14)
        plt.tight_layout()
        plt.rcParams['font.sans-serif'] = 'Arial'
        if save_dir is not None:
            sensor_name_str = '_'.join(self.sensor_names)
            fig.savefig(f'{save_dir}/{sensor_name_str}_plot_time_series.svg', format='svg', dpi=save_dpi, transparent=True, bbox_inches='tight')
        else:
            plt.show()

    def plot_single_roi(self, full_array_column, session_id, roi, selected_stim_ids=None, fig_size=(6.5, 8), dpi=300, y_limits=None, save_dir=None, save_dpi=300):
        """
        Plots the time series data for a specific ROI within a given session for selected stimulation IDs.
        :param full_array_column: The name of the column with time series data.
        :param session_id: The ID of the session to filter.
        :param roi: The specific ROI to plot.
        :param selected_stim_ids: List of stimulation IDs to plot. If None, plot all available stimulations.
        :param fig_size: Tuple representing the figure size of the plot in inches (default: 6.5 x 8).
        :param dpi: The resolution in dots per inch for displaying the figure (default: 300).
        :param y_limits: Tuple representing the y-axis min and max (y_min, y_max). If None, use default auto-scaling.
        :param save_dir: Directory to save the plot. If None, the plot is not saved.
        :param save_dpi: The resolution in dots per inch for saving the figure (default: 300).
        """
        if self.combined_df is None:
            raise ValueError('Data has not been prepared for plotting. Call prepare_for_plotting first.')
        if session_id not in self.combined_df['session_id'].unique():
            print(f'Error: Session ID {session_id} does not exist in the dataset.')
            print(f"Available session IDs: {self.combined_df['session_id'].unique()}")
            return
        session_data = self.combined_df[self.combined_df['session_id'] == session_id]
        if roi not in session_data['roi'].unique():
            print(f'Error: ROI {roi} does not exist for Session ID {session_id}.')
            print(f"Available ROIs for session {session_id}: {session_data['roi'].unique()}")
            return
        print(f'Plotting data for Session ID: {session_id}, ROI: {roi}')
        filtered_data = session_data[session_data['roi'] == roi]
        if filtered_data.empty:
            raise ValueError(f'No data found for session_id: {session_id} and roi: {roi}')
        stim_ids = selected_stim_ids if selected_stim_ids is not None else filtered_data['stimulation_id'].unique()
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        for stim_id in stim_ids:
            stim_data = filtered_data[filtered_data['stimulation_id'] == stim_id]
            if stim_data.empty:
                print(f'No data found for Stimulation ID: {stim_id}')
                continue
            row = stim_data.iloc[0]
            sample_size = len(row[full_array_column])
            time_vector = (np.arange(sample_size) - 10) * 100
            ax.plot(time_vector, row[full_array_column], label=f'Stim ID: {stim_id}')
        ax.set_title(f'Session {session_id} - ROI {roi}', fontsize=14)
        ax.set_xlabel('Time (ms)', fontsize=18)
        ax.set_ylabel('ΔF/F', fontsize=18)
        if y_limits is not None:
            ax.set_ylim(y_limits)
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        ax.legend()
        plt.tight_layout()
        plt.rcParams['font.sans-serif'] = 'Arial'
        if save_dir is not None:
            mpl.rcParams['svg.fonttype'] = 'none'
            fig.savefig(f'{save_dir}/session_{session_id}_ROI_{roi}.svg', format='svg', dpi=save_dpi, transparent=True, bbox_inches='tight')

    def plot_non_responsive_heatmap_and_pie(self, selected_stim_id, fig_size=(15, 6), dpi=300, cmap='rocket', vmin=0, vmax=None, smooth_method=None, smooth_sigma=1, save_dir=None, save_dpi=300, interpolation='gaussian'):
        """
        Creates a 1x2 panel with a heatmap of traces and a pie chart of responsive vs. non-responsive ROIs.
        
        :param selected_stim_id: The stimulation ID to analyze.
        :param fig_size: Tuple representing the figure size (width, height) in inches.
        :param dpi: The resolution in dots per inch for display.
        :param cmap: Colormap to use. Default is 'rocket'.
        :param vmin: Minimum value for colormap scaling. Default is 0.
        :param vmax: Maximum value for colormap scaling. If None, it's set to the data maximum.
        :param smooth_method: Method for smoothing. Options are 'gaussian', 'moving_average', or None. Default is None (no smoothing).
        :param smooth_sigma: Sigma for Gaussian filter or window size for moving average. Default is 1.
        :param save_dir: Directory to save the plot. If None, the plot is not saved.
        :param save_dpi: The resolution in dots per inch for saving the figure (default: 300).
        :param interpolation: Interpolation method for imshow. Options include 'nearest', 'bilinear', 'bicubic', 'spline16',
                          'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 
                          'bessel', 'mitchell', 'sinc', 'lanczos'. Default is 'gaussian'.
        """
        df = self.data_frames[0]
        stim_data = df[df['stimulation_id'] == selected_stim_id]

        def process_array(arr):
            arr = np.array(arr)
            arr[np.isnan(arr) | (arr < 0)] = 0
            return arr
        stim_data['processed_array'] = stim_data['delta_f_f_full_array'].apply(process_array)
        responsive_data = stim_data[stim_data['is_responsive']]
        non_responsive_data = stim_data[~stim_data['is_responsive']]
        responsive_data['max_delta_f'] = responsive_data['processed_array'].apply(np.max)
        responsive_data = responsive_data.sort_values('max_delta_f', ascending=True)
        combined_data = pd.concat([responsive_data, non_responsive_data])

        def smooth_data(data, method=None, sigma=1):
            if method == 'gaussian':
                return gaussian_filter1d(data, sigma=sigma)
            elif method == 'moving_average':
                window = np.ones(int(sigma)) / float(sigma)
                return np.convolve(data, window, 'same')
            else:
                return data
        if smooth_method is not None:
            combined_data['plot_array'] = combined_data['processed_array'].apply(lambda x: smooth_data(x, method=smooth_method, sigma=smooth_sigma))
        else:
            combined_data['plot_array'] = combined_data['processed_array']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size, dpi=dpi)
        heatmap_data = np.vstack(combined_data['processed_array'].values)
        if vmax is None:
            vmax = np.nanmax(heatmap_data)
        base_cmap = plt.get_cmap(cmap)
        colors = base_cmap(np.linspace(0, 1, 256))
        colors[0] = [0, 0, 0, 1]
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        im = ax1.imshow(heatmap_data, cmap=custom_cmap, aspect='auto', interpolation=interpolation, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax1, label='dF/F')
        ax1.set_title(f'Neuron Traces (Stim ID: {selected_stim_id})')
        ax1.set_ylabel('Neuron #')
        num_neurons = len(combined_data)
        ax1.set_yticks([0, num_neurons - 1])
        ax1.set_yticklabels([1, num_neurons])
        ax1.axvline(x=10, color='white', linestyle='--', linewidth=2)
        num_frames = heatmap_data.shape[1]
        time_points = np.linspace(-1, (num_frames - 11) / 10, num_frames)
        ax1.set_xticks(np.linspace(0, num_frames - 1, 5))
        ax1.set_xticklabels([f'{t:.1f}' for t in np.linspace(time_points[0], time_points[-1], 5)])
        ax1.set_xlabel('Time relative to stimulus onset (s)')
        responsive_count = len(responsive_data)
        ax1.axhline(y=responsive_count, color='white', linestyle=':', linewidth=2)
        sizes = [len(responsive_data), len(non_responsive_data)]
        labels = ['Responsive', 'Non-responsive']
        colors = ['#ccccff', '#9933ff']
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'ROI Responsiveness (Stim ID: {selected_stim_id})')
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.rcParams['svg.fonttype'] = 'none'
            sensor_name = self.sensor_names[0] if self.sensor_names else 'unknown_sensor'
            filename = f'{save_dir}/{sensor_name}_heatmap_pie_stim_{selected_stim_id}.svg'
            fig.savefig(filename, format='svg', dpi=save_dpi, transparent=True, bbox_inches='tight')
            print(f'Figure saved as {filename}')
        plt.show()

    def plot_cumulative_distribution(self, column_name='time_to_peak', selected_stim_ids=None, fig_size=(10, 6), dpi=300, save_dir=None, save_dpi=300, num_points=1000):
        """
        Plots the cumulative distribution of a specified column (default: time_to_peak) for each sensor.
        
        :param column_name: The name of the column to plot (default: 'time_to_peak')
        :param selected_stim_ids: List of stimulation IDs to include. If None, use all.
        :param fig_size: Tuple representing the figure size (width, height) in inches.
        :param dpi: The resolution in dots per inch for display.
        :param save_dir: Directory to save the plot. If None, the plot is not saved.
        :param save_dpi: The resolution in dots per inch for saving the figure.
        :param num_points: Number of points to use for the cumulative distribution (default: 1000)
        """
        if self.combined_df is None:
            self.prepare_for_plotting(column_name)
        if selected_stim_ids is not None:
            plot_df = self.combined_df[self.combined_df['stimulation_id'].isin(selected_stim_ids)]
        else:
            plot_df = self.combined_df
        plt.figure(figsize=fig_size, dpi=dpi)
        for sensor_name in self.sensor_names:
            sensor_data = plot_df[plot_df['sensor_name'] == sensor_name][column_name]
            print(f'Data for {sensor_name}:')
            print(f'  Number of data points: {len(sensor_data)}')
            print(f'  Min value: {sensor_data.min()}')
            print(f'  Max value: {sensor_data.max()}')
            print(f'  Mean value: {sensor_data.mean()}')
            print(f'  Median value: {sensor_data.median()}')
            sorted_data = np.sort(sensor_data)
            x = np.linspace(sorted_data.min(), sorted_data.max(), num_points)
            y = np.zeros_like(x)
            for i, value in enumerate(x):
                y[i] = np.sum(sorted_data <= value) / len(sorted_data) * 100
            plt.vlines(x[0], 0, y[0], color=self.sensor_box_colors[sensor_name], linestyle='-')
            plt.plot(x, y, label=sensor_name, color=self.sensor_box_colors[sensor_name])
        plt.xlabel(column_name, fontsize=14)
        plt.ylabel('Cumulative Percentage', fontsize=14)
        plt.title(f'Cumulative Distribution of {column_name}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(False)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.rcParams['svg.fonttype'] = 'none'
            sensor_names = '_'.join(self.sensor_names)
            stim_ids = '_'.join(map(str, selected_stim_ids)) if selected_stim_ids else 'all'
            filename = f'{save_dir}/{sensor_names}_{column_name}_cumulative_dist_stim_{stim_ids}.svg'
            plt.savefig(filename, format='svg', dpi=save_dpi, transparent=True, bbox_inches='tight')
            print(f'Figure saved as {filename}')
        plt.show()

    def run_ks_test(self, column_name='time_to_peak', selected_stim_ids=None):
        """
        Runs a two-tailed Kolmogorov-Smirnov test on the specified column for each pair of sensors.
        
        :param column_name: The name of the column to analyze (default: 'time_to_peak')
        :param selected_stim_ids: List of stimulation IDs to include. If None, use all.
        :return: DataFrame with test results
        """
        if self.combined_df is None:
            self.prepare_for_plotting(column_name)
        if selected_stim_ids is not None:
            plot_df = self.combined_df[self.combined_df['stimulation_id'].isin(selected_stim_ids)]
        else:
            plot_df = self.combined_df
        results = []
        for i in range(len(self.sensor_names)):
            for j in range(i + 1, len(self.sensor_names)):
                sensor1 = self.sensor_names[i]
                sensor2 = self.sensor_names[j]
                data1 = plot_df[plot_df['sensor_name'] == sensor1][column_name]
                data2 = plot_df[plot_df['sensor_name'] == sensor2][column_name]
                ks_statistic, p_value = stats.ks_2samp(data1, data2)
                results.append({'Sensor1': sensor1, 'Sensor2': sensor2, 'N1': len(data1), 'N2': len(data2), 'KS_statistic': ks_statistic, 'p_value': p_value, 'Column': column_name, 'Stim_IDs': ', '.join(map(str, selected_stim_ids)) if selected_stim_ids else 'All'})
        results_df = pd.DataFrame(results)

        def get_significance(p):
            if p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            else:
                return 'ns'
        results_df['Significance'] = results_df['p_value'].apply(get_significance)
        results_df['p_value_formatted'] = results_df['p_value'].apply(lambda p: f'{p:.4f}' if p >= 0.0001 else '<0.0001')
        column_order = ['Sensor1', 'Sensor2', 'N1', 'N2', 'KS_statistic', 'p_value', 'p_value_formatted', 'Significance', 'Column', 'Stim_IDs']
        results_df = results_df[column_order]
        return results_df

    def plot_mean_with_error(self, df_column_name, error_type='SEM', selected_stim_ids=None, xlim=None, ylim=None, fig_size=(8, 6), dpi=300, save_dir=None, save_dpi=300):
        """
        Plots the mean values with error bars for selected stimulation IDs across sensors, using consistent colors and a logarithmic x-axis.
        :param df_column_name: The name of the column to use for the value in the plot.
        :param error_type: The type of error to display ('SD' for Standard Deviation or 'SEM' for Standard Error of the Mean).
        :param selected_stim_ids: List of stimulation IDs to plot. If None, plot all.
        :param xlim: Tuple representing the x-axis limits (min, max). If None, use default.
        :param ylim: Tuple representing the y-axis limits (min, max). If None, use default.
        :param fig_size: Tuple representing the figure size (width, height) in inches.
        :param dpi: The resolution in dots per inch.
        :param save_dir: Directory to save the plot. If None, the plot is not saved.
        :param save_dpi: The resolution in dots per inch for saving the figure.
        """
        if self.combined_df is None:
            self.prepare_for_plotting(df_column_name)
        if selected_stim_ids is not None:
            plot_df = self.combined_df[self.combined_df['stimulation_id'].isin(selected_stim_ids)]
        else:
            plot_df = self.combined_df
            selected_stim_ids = sorted(plot_df['stimulation_id'].unique())
        plt.figure(figsize=fig_size, dpi=dpi)
        ax = plt.subplot()
        for sensor_name in self.sensor_names:
            sensor_data = plot_df[plot_df['sensor_name'] == sensor_name]
            means = sensor_data.groupby('stimulation_id')[df_column_name].mean()
            if error_type == 'SD':
                errors = sensor_data.groupby('stimulation_id')[df_column_name].std()
            else:
                errors = sensor_data.groupby('stimulation_id')[df_column_name].sem()
            ax.errorbar(means.index, means, yerr=errors, label=sensor_name, fmt='-o', capsize=5, color=self.sensor_box_colors[sensor_name])
        ax.set_xscale('log')
        ax.set_xticks(selected_stim_ids)
        ax.set_xticklabels(selected_stim_ids)
        if xlim:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(min(selected_stim_ids) * 0.8, max(selected_stim_ids) * 1.2)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlabel('Stimulation ID', fontsize=14)
        ax.set_ylabel(df_column_name, fontsize=14)
        ax.set_title(f'Mean {df_column_name} by Stimulation ID across Sensors', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(title='Sensor', loc='best', fontsize=10)
        ax.grid(False)
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.rcParams['svg.fonttype'] = 'none'
            sensor_names = '_'.join(self.sensor_names)
            stim_ids = '_'.join(map(str, selected_stim_ids))
            filename = f'{save_dir}/{sensor_names}_{df_column_name}_mean_error_stim_{stim_ids}.svg'
            plt.savefig(filename, format='svg', dpi=save_dpi, transparent=True, bbox_inches='tight')
            print(f'Figure saved as {filename}')
        plt.show()

    def analyze_responsive_neurons_by_session(self, stim_ids):
        """
        Analyzes the number and percentage of responsive neurons for specified stimulation IDs,
        grouped by session and sensor type.
        
        Parameters:
        -----------
        stim_ids : int or list
            Single stimulation ID or list of stimulation IDs to analyze
        
        Returns:
        --------
        dict
            Nested dictionary containing analysis results organized by:
            sensor -> session -> stim_id -> {total_neurons, responsive_neurons, percent_responsive}
        pandas.DataFrame
            Summary DataFrame containing the analysis results in a tabular format
        """
        if isinstance(stim_ids, (int, float)):
            stim_ids = [stim_ids]
        results = {sensor: {} for sensor in self.sensor_names}
        summary_rows = []
        for sensor_idx, sensor_name in enumerate(self.sensor_names):
            df = self.data_frames[sensor_idx]
            unique_sessions = df['session_id'].unique()
            for session_id in unique_sessions:
                session_data = df[df['session_id'] == session_id]
                results[sensor_name][session_id] = {}
                for stim_id in stim_ids:
                    stim_data = session_data[session_data['stimulation_id'] == stim_id]
                    if not stim_data.empty:
                        total_neurons = len(stim_data)
                        responsive_neurons = stim_data['is_responsive'].sum()
                        percent_responsive = responsive_neurons / total_neurons * 100
                        results[sensor_name][session_id][stim_id] = {'total_neurons': total_neurons, 'responsive_neurons': responsive_neurons, 'percent_responsive': percent_responsive}
                        summary_rows.append({'sensor': sensor_name, 'session_id': session_id, 'stim_id': stim_id, 'total_neurons': total_neurons, 'responsive_neurons': responsive_neurons, 'percent_responsive': percent_responsive})
        summary_df = pd.DataFrame(summary_rows)
        summary_stats = summary_df.groupby(['sensor', 'stim_id']).agg({'percent_responsive': ['mean', 'std', 'count']}).round(2)
        print('\nSummary Statistics for Responsive Neurons:')
        print('=========================================')
        for sensor in self.sensor_names:
            print(f'\n{sensor}:')
            for stim_id in stim_ids:
                stats = summary_stats.loc[sensor, stim_id]['percent_responsive']
                print(f'\nStim ID {stim_id}:')
                print(f"  Mean % Responsive: {stats['mean']:.2f}%")
                print(f"  Std Dev: {stats['std']:.2f}%")
                print(f"  Number of Sessions: {stats['count']}")
        return (results, summary_df)

    def analyze_responsive_neurons_by_fov(self, stim_ids, n_fovs_per_session=4):
        """
        Analyzes the number and percentage of responsive neurons by subdividing each session
        into virtual Fields of View (FOVs), then calculating statistics for each FOV.
        
        Parameters:
        -----------
        stim_ids : int or list
            Single stimulation ID or list of stimulation IDs to analyze
        n_fovs_per_session : int
            Number of virtual FOVs to create per session (default: 4)
        
        Returns:
        --------
        dict
            Nested dictionary containing analysis results organized by:
            sensor -> session -> fov -> stim_id -> {total_neurons, responsive_neurons, percent_responsive}
        pandas.DataFrame
            Summary DataFrame containing the analysis results in a tabular format
        """
        if isinstance(stim_ids, (int, float)):
            stim_ids = [stim_ids]
        results = {sensor: {} for sensor in self.sensor_names}
        summary_rows = []
        for sensor_idx, sensor_name in enumerate(self.sensor_names):
            df = self.data_frames[sensor_idx]
            unique_sessions = df['session_id'].unique()
            for session_id in unique_sessions:
                session_data = df[df['session_id'] == session_id]
                unique_rois = session_data['roi'].unique()
                np.random.seed(42)
                shuffled_rois = np.random.permutation(unique_rois)
                roi_groups = np.array_split(shuffled_rois, n_fovs_per_session)
                results[sensor_name][session_id] = {}
                for fov_idx, fov_rois in enumerate(roi_groups):
                    fov_id = f'{session_id}_FOV{fov_idx + 1}'
                    results[sensor_name][session_id][fov_id] = {}
                    fov_data = session_data[session_data['roi'].isin(fov_rois)]
                    for stim_id in stim_ids:
                        stim_data = fov_data[fov_data['stimulation_id'] == stim_id]
                        if not stim_data.empty:
                            total_neurons = len(stim_data)
                            responsive_neurons = stim_data['is_responsive'].sum()
                            percent_responsive = responsive_neurons / total_neurons * 100
                            results[sensor_name][session_id][fov_id][stim_id] = {'total_neurons': total_neurons, 'responsive_neurons': responsive_neurons, 'percent_responsive': percent_responsive}
                            summary_rows.append({'sensor': sensor_name, 'session_id': session_id, 'fov_id': fov_id, 'stim_id': stim_id, 'total_neurons': total_neurons, 'responsive_neurons': responsive_neurons, 'percent_responsive': percent_responsive})
        summary_df = pd.DataFrame(summary_rows)
        summary_stats = summary_df.groupby(['sensor', 'stim_id']).agg({'percent_responsive': ['mean', 'std', 'count'], 'total_neurons': ['mean', 'sum']}).round(2)
        print('\nSummary Statistics for Responsive Neurons (FOV Analysis):')
        print('=====================================================')
        for sensor in self.sensor_names:
            print(f'\n{sensor}:')
            for stim_id in stim_ids:
                stats = summary_stats.loc[sensor, stim_id]
                print(f'\nStim ID {stim_id}:')
                print(f"  Mean % Responsive: {stats['percent_responsive']['mean']:.2f}%")
                print(f"  Std Dev: {stats['percent_responsive']['std']:.2f}%")
                print(f"  Number of FOVs: {stats['percent_responsive']['count']}")
                print(f"  Mean Neurons per FOV: {stats['total_neurons']['mean']:.1f}")
                print(f"  Total Neurons: {stats['total_neurons']['sum']}")
        return (results, summary_df)

    def run_kruskal_wallis_test(self, summary_df, stim_ids=None):
        """
        Performs Kruskal-Wallis test for comparing three independent sensors, followed by
        post-hoc Mann-Whitney U tests with Bonferroni correction.
        
        Parameters:
        -----------
        summary_df : pandas.DataFrame
            Summary DataFrame from either analyze_responsive_neurons_by_session or analyze_responsive_neurons_by_fov
        stim_ids : list or None
            List of stimulation IDs to analyze. If None, uses all in the data
        
        Returns:
        --------
        dict
            Dictionary containing Kruskal-Wallis results and post-hoc test results for each stimulation ID
        """
        if len(self.sensor_names) != 3:
            raise ValueError('This analysis requires exactly three sensors.')
        if stim_ids is None:
            stim_ids = sorted(summary_df['stim_id'].unique())
        elif isinstance(stim_ids, (int, float)):
            stim_ids = [stim_ids]
        sensor_pairs = [(self.sensor_names[i], self.sensor_names[j]) for i in range(len(self.sensor_names)) for j in range(i + 1, len(self.sensor_names))]
        results = {}
        for stim_id in stim_ids:
            print(f'\nAnalysis for Stimulation ID {stim_id}')
            print('=' * 40)
            stim_data = summary_df[summary_df['stim_id'] == stim_id]
            sensor_data = {sensor: stim_data[stim_data['sensor'] == sensor]['percent_responsive'].values for sensor in self.sensor_names}
            results[stim_id] = {'descriptive_stats': {}, 'kruskal_wallis': {}, 'post_hoc': {}}
            print('\nDescriptive Statistics:')
            print('-----------------')
            for sensor in self.sensor_names:
                desc_stats = {'n': len(sensor_data[sensor]), 'mean': np.mean(sensor_data[sensor]), 'median': np.median(sensor_data[sensor]), 'std': np.std(sensor_data[sensor], ddof=1), 'sem': stats.sem(sensor_data[sensor])}
                results[stim_id]['descriptive_stats'][sensor] = desc_stats
                print(f'\n{sensor}:')
                print(f"  N: {desc_stats['n']}")
                print(f"  Mean: {desc_stats['mean']:.2f}%")
                print(f"  Median: {desc_stats['median']:.2f}%")
                print(f"  Std Dev: {desc_stats['std']:.2f}%")
                print(f"  SEM: {desc_stats['sem']:.2f}%")
            print('\nKruskal-Wallis Test:')
            print('-----------------')
            all_data = []
            groups = []
            for sensor in self.sensor_names:
                all_data.extend(sensor_data[sensor])
                groups.extend([sensor] * len(sensor_data[sensor]))
            h_stat, kw_p = stats.kruskal(*[sensor_data[sensor] for sensor in self.sensor_names])
            results[stim_id]['kruskal_wallis'] = {'h_statistic': h_stat, 'p_value': kw_p}
            print(f'H-statistic: {h_stat:.3f}')
            print(f'p-value: {kw_p:.4f}')
            print(f'Significant difference: {kw_p < 0.05}')
            if kw_p < 0.05:
                print('\nPost-hoc Analysis (Mann-Whitney U with Bonferroni correction):')
                print('-------------------------------------------------------')
                n_comparisons = len(sensor_pairs)
                alpha_bonferroni = 0.05 / n_comparisons
                pairwise_results = []
                for sensor1, sensor2 in sensor_pairs:
                    data1 = sensor_data[sensor1]
                    data2 = sensor_data[sensor2]
                    stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    n1, n2 = (len(data1), len(data2))
                    z_score = (stat - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                    effect_size = np.abs(z_score) / np.sqrt(n1 + n2)
                    pairwise_results.append({'pair': (sensor1, sensor2), 'statistic': stat, 'p_value': p_val, 'significant': p_val < alpha_bonferroni, 'effect_size': effect_size})
                    print(f'\n{sensor1} vs {sensor2}:')
                    print(f'  Mann-Whitney U statistic: {stat:.3f}')
                    print(f'  p-value: {p_val:.4f}')
                    print(f'  Significant (α={alpha_bonferroni:.4f}): {p_val < alpha_bonferroni}')
                    print(f'  Effect size (r): {effect_size:.3f}')
                    print(f'  {sensor1} median: {np.median(data1):.2f}%, mean: {np.mean(data1):.2f}%')
                    print(f'  {sensor2} median: {np.median(data2):.2f}%, mean: {np.mean(data2):.2f}%')
                results[stim_id]['post_hoc'] = {'method': 'mann_whitney_u', 'alpha_bonferroni': alpha_bonferroni, 'results': pairwise_results}
        return results

    def plot_fov_responsiveness_with_stats(self, summary_df, kw_results, stim_id, box_width=0.8, strip_size=3, fig_size=(10, 6), dpi=300, save_dir=None, save_dpi=300):
        """
        Creates a box plot showing the distribution of responsive neuron percentages
        across FOVs for each sensor with consistent y-axis scaling.
        
        Parameters:
        -----------
        summary_df : pandas.DataFrame
            Summary DataFrame from analyze_responsive_neurons_by_fov
        kw_results : dict
            Results dictionary from run_kruskal_wallis_test
        stim_id : int
            Stimulation ID to plot
        box_width : float
            Width of the boxplots
        strip_size : int
            Size of the points in the stripplot
        fig_size : tuple
            Figure size (width, height) in inches
        dpi : int
            Resolution for display
        save_dir : str or None
            Directory to save the plot. If None, the plot is not saved
        save_dpi : int
            Resolution for saved figure
        """
        stim_data = summary_df[summary_df['stim_id'] == stim_id]
        plt.figure(figsize=fig_size, dpi=dpi)
        ax = plt.subplot()
        boxprops = {'edgecolor': 'k', 'linewidth': 1.5}
        lineprops = {'color': 'k', 'linewidth': 1.5}
        boxplot_kwargs = {'boxprops': boxprops, 'medianprops': lineprops, 'whiskerprops': lineprops, 'capprops': lineprops, 'width': box_width, 'palette': self.sensor_box_colors}
        stripplot_kwargs = {'linewidth': 0.1, 'size': strip_size, 'alpha': 0.3, 'palette': self.sensor_strip_colors}
        sns.stripplot(x='sensor', y='percent_responsive', data=stim_data, jitter=0.2, **stripplot_kwargs)
        sns.boxplot(x='sensor', y='percent_responsive', data=stim_data, fliersize=0, **boxplot_kwargs)
        ax.set_ylim(-5, 110)
        ax.set_xlabel('Sensor', fontsize=18)
        ax.set_ylabel('Percent Responsive', fontsize=18)
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        ax.yaxis.grid(False, linestyle='--', alpha=0.3)
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.tight_layout()
        print(f'\nStatistical Results for Stim ID {stim_id}:')
        print('-' * 40)
        print(f'Kruskal-Wallis test:')
        print(f"H-statistic: {kw_results[stim_id]['kruskal_wallis']['h_statistic']:.3f}")
        print(f"p-value: {kw_results[stim_id]['kruskal_wallis']['p_value']:.4f}")
        if kw_results[stim_id]['kruskal_wallis']['p_value'] < 0.05:
            print('\nPost-hoc Mann-Whitney U tests (Bonferroni-corrected):')
            for result in kw_results[stim_id]['post_hoc']['results']:
                print(f"\n{result['pair'][0]} vs {result['pair'][1]}:")
                print(f"p-value: {result['p_value']:.4f}")
                print(f"Significant: {result['significant']}")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.rcParams['svg.fonttype'] = 'none'
            sensor_names = '_'.join(self.sensor_names)
            filename = f'{save_dir}/{sensor_names}_FOV_responsiveness_stim_{stim_id}.svg'
            plt.savefig(filename, format='svg', dpi=save_dpi, transparent=True, bbox_inches='tight')
            print(f'\nFigure saved as {filename}')
        plt.show()

def count_and_list_rois_per_stim(df, stim_ids=None, include='both'):
    """
    Counts and lists the ROIs being plotted for each stimulation condition, optionally filtering by responsiveness.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the responsiveness data.
    stim_ids : list, optional
        List of stimulation IDs to filter on. If None, all unique IDs in the DataFrame will be used.
    include : str, optional
        Filter for 'responsive', 'non-responsive', or 'both' units.

    Returns
    -------
    dict
        A dictionary where each key is a stimulation ID and the value is a dictionary containing:
            - 'num_rois': The number of ROIs plotted for that stimulation.
            - 'rois': A list of the ROIs being plotted for that stimulation.
    """
    if stim_ids is None:
        stim_ids = sorted(df['stimulation_id'].unique())
    else:
        stim_ids = sorted(stim_ids)
    rois_per_stim = {}
    for stim_id in stim_ids:
        stim_df = df[df['stimulation_id'] == stim_id]
        if include != 'both':
            stim_df = stim_df[stim_df['is_responsive'] == (include == 'responsive')]
        unique_rois = stim_df['roi'].unique()
        rois_per_stim[stim_id] = {'num_rois': len(unique_rois), 'rois': unique_rois.tolist()}
    return rois_per_stim

def filter_responsive_rois_by_stimulation_correction(session_data, responsiveness_df, valid_responsive_rois_by_session):
    """
    Filters session data for responsive ROIs corrected by background removal and ensures only valid ROIs are processed.

    Parameters
    ----------
    session_data : dict
        Dictionary where keys are session IDs and values are dataframes containing the calcium signals for each session.
    responsiveness_df : pd.DataFrame
        Dataframe containing information about responsive ROIs, with columns such as 'session_id', 'roi', 'is_responsive', 
        and 'stimulation_id'.
    valid_responsive_rois_by_session : dict
        Dictionary where keys are session IDs and values are lists of valid ROIs that still exist after correction.

    Returns
    -------
    dict
        A dictionary where each key is a session ID and the value is a dataframe with the filtered responsive ROIs.
    """
    filtered_data_by_session = {}
    responsive_df = responsiveness_df[responsiveness_df['is_responsive'] & (responsiveness_df['stimulation_id'] == 12)]
    grouped_responsive_df = responsive_df.groupby('session_id')
    for session_id, group in grouped_responsive_df:
        session_frames_list = []
        unique_rois = group['roi'].unique()
        session_df = session_data.get(session_id)
        if session_df is None:
            print(f'Session ID {session_id} not found in session_data.')
            continue
        valid_rois = valid_responsive_rois_by_session.get(session_id, [])
        for roi in unique_rois:
            if roi in valid_rois:
                roi_number = re.search('\\d+', roi)
                if not roi_number:
                    print(f'ROI format is incorrect for {roi}')
                    continue
                roi_column_name = f'ROI_{roi_number.group()}'
                if roi_column_name in session_df.columns:
                    roi_frames_df = session_df[[roi_column_name]].copy()
                    session_frames_list.append(roi_frames_df)
                else:
                    print(f'Column {roi_column_name} not found in session dataframe for session_id {session_id}.')
        if session_frames_list:
            combined_frames_df = pd.concat(session_frames_list, axis=1)
            filtered_data_by_session[session_id] = combined_frames_df
    return filtered_data_by_session

def prepare_responsive_rois(session_data, responsiveness_df):
    """
    Identifies valid responsive ROIs for each session, ensuring that only the ROIs still present in the session data
    after background correction are included.

    Parameters
    ----------
    session_data : dict
        A dictionary where keys are session IDs and values are dataframes containing the calcium signals for each session.
        The dataframes should include ROI columns (e.g., 'ROI_1', 'ROI_2', etc.) and any other metadata.
    responsiveness_df : pd.DataFrame
        A dataframe containing information about responsive ROIs. It must include columns such as 'session_id', 'roi',
        'is_responsive', and 'stimulation_id'. This dataframe is used to identify which ROIs are considered responsive
        based on stimulation.

    Returns
    -------
    dict
        A dictionary where each key is a session ID and the value is a list of valid responsive ROIs for that session.
        The valid ROIs are those that are marked as responsive and still exist in the corresponding session data after
        correction.

    Notes
    -----
    - The function filters ROIs based on their responsiveness to a specific stimulation (e.g., stimulation_id == 12).
    - Only ROIs that are marked as responsive and still exist in the session dataframe after background correction
      will be returned.
    - This method is designed to handle the situation where some ROIs have been permanently removed due to background
      correction in previous steps, and ensures downstream processes only use valid ROIs.

    Example
    -------
    valid_responsive_rois = prepare_responsive_rois(all_data_gcamp8_session_data, responsiveness_df_gcamp8)
    """
    valid_responsive_rois_by_session = {}
    responsive_df = responsiveness_df[responsiveness_df['is_responsive'] & (responsiveness_df['stimulation_id'] == 12)]
    grouped_responsive_df = responsive_df.groupby('session_id')
    for session_id, group in grouped_responsive_df:
        session_df = session_data.get(session_id)
        if session_df is None:
            print(f'Session ID {session_id} not found in session_data.')
            continue
        unique_rois = group['roi'].unique()
        print(f'Session {session_id}: Responsive ROIs in responsiveness_df: {unique_rois}')
        print(f'Session {session_id}: Available ROI columns in session_df: {session_df.columns}')
        valid_rois = [roi for roi in unique_rois if roi in session_df.columns]
        if not valid_rois:
            print(f'No valid responsive ROIs found in session {session_id}.')
            continue
        valid_responsive_rois_by_session[session_id] = valid_rois
    return valid_responsive_rois_by_session

def save_data_to_csv(combined_df, df_column_name, selected_stim_ids, sensor_names, save_dir):
    """
    Saves the filtered data as a CSV file for selected stimulation IDs.
    :param combined_df: The combined DataFrame containing the data.
    :param df_column_name: The name of the column to use for the value in the CSV.
    :param selected_stim_ids: List of stimulation IDs to save. If None, save all.
    :param sensor_names: List of sensor names used in the data.
    :param save_dir: Directory to save the CSV file.
    """
    import os
    import pandas as pd
    df_column_name = str(df_column_name)
    if selected_stim_ids is not None:
        if not isinstance(selected_stim_ids, list):
            raise ValueError('The selected_stim_ids parameter must be a list of stimulation IDs.')
        if not all((isinstance(stim_id, int) for stim_id in selected_stim_ids)):
            raise ValueError('All elements in the selected_stim_ids list must be integers.')
    if selected_stim_ids is not None:
        filtered_df = combined_df[combined_df['stimulation_id'].isin(selected_stim_ids)]
    else:
        filtered_df = combined_df.copy()
    if filtered_df.empty:
        raise ValueError('No data available for the selected stimulation IDs.')
    os.makedirs(save_dir, exist_ok=True)
    sensor_names_str = '_'.join(sensor_names)
    if selected_stim_ids is not None:
        stim_ids_str = '_'.join(map(str, selected_stim_ids))
    else:
        stim_ids_str = 'all'
    filename = f'{save_dir}/{sensor_names_str}_{df_column_name}_stim_{stim_ids_str}.csv'
    filtered_df.to_csv(filename, index=False)
    print(f'Data saved as {filename}')
