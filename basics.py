from scipy import signal
from torch.utils.data import Dataset
import numpy as np

def bandpass(data, fs, fc_low=5, fc_high=20):
        """
        Apply a bandpass filter onto the signal, and save the filtered
        signal.
        """
    

        b, a = signal.butter(2, [float(fc_low) * 2 / fs,
                                 float(fc_high) * 2 / fs], 'pass')
        return signal.filtfilt(b, a, data,
                                     axis=0)
    
class FilteredDataset(Dataset):
    def __init__(self, data, low_cut = 5, high_cut = 20):
        
        self.data = data.values
        self.no_classes = 4
        self.low_cut = low_cut
        self.high_cut = high_cut
        self._filter_signal()
        
    def _apply_bandpass(self, row):
        return [bandpass(row[i : i + 5000], 500, self.low_cut, self.high_cut) for i in range(2, 55003, 5000)]
        
    def _filter_signal(self):
        self.filtered_data = np.apply_along_axis(lambda x: self._apply_bandpass(x),1, self.data)

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return len(self.filtered_data)
    
    def _get_label(self, idx):
        label = np.zeros((self.no_classes), dtype=np.float32)
        label[int(self.data[idx][-1])] = 1
        return label

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        
        return self.filtered_data[idx], self._get_label(idx)
    
    
class FixedLengthDataset(FilteredDataset):
    def __init__(self, data, peaks, signal_length, min_start_offset = 50, max_start_offset = 300, low_cut = 5, high_cut = 20):
        super().__init__(data, low_cut, high_cut)
        self.peaks = peaks.values
        self.signal_length = signal_length
        self.min_start_offset = min_start_offset
        self.max_start_offset = max_start_offset
        self._get_cuts()
        
    def _get_peaks_for_signal_cuts(self, peaks):
        try:
            filtered_peaks = [
                peak for peak in peaks 
                if peak < 5000 - self.signal_length or 5000 - self.signal_length > max(0, peak - self.max_start_offset) 
                             ]
            return filtered_peaks
        except:
            print("Exception: ", peaks)
            return []

    def _get_cuts(self):
        cut_points = []
        for i, peaks in enumerate(self.peaks):
            filtered_peaks =  self._get_peaks_for_signal_cuts(peaks[0]) # only for first (i) led
            for point in filtered_peaks:
                cut_points.append([i, point])
        self.cut_points = cut_points
        
    def _cut(self, rows, cut_point):
        start_offset =  cut_point - np.random.randint(self.min_start_offset, self.max_start_offset+1)
        start_offset = min(start_offset, 5000 - self.signal_length)
        start_offset = int(max(0, start_offset))
        end_offset = start_offset + self.signal_length
        try:
            cut_rows = [row[start_offset:end_offset] for row in rows]
            return np.array(cut_rows)
        except:
            print ("Cut Exception:")
            print(rows)
            print(cut_point, start_offset, end_offset)
            return np.array([0])
        
    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return len(self.cut_points)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        true_idx, cut_point = self.cut_points[idx]
        label = self._get_label(true_idx)
        cut_rows = self._cut(self.filtered_data[true_idx], cut_point)
        
        return cut_rows, label