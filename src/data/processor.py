import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from config.settings import PROCESSED_DATA_DIR


class EEGPreprocessor:
    """
    EEG preprocessing pipeline
    - Robust .mat unwrapping
    - Non-EEG column detection & removal
    - Sampling-rate detection heuristic and bandpass filtering
    - Feature extraction and saving
    """
    
    def __init__(self, sampling_rate: int = 256, lowcut: float = 1.0, highcut: float = 50.0):
        self.sampling_rate = sampling_rate  # default if detection fails
        self.lowcut = lowcut
        self.highcut = highcut
        self.scaler = StandardScaler()
    
    def detect_sampling_rate(self, df: pd.DataFrame) -> Optional[int]:
        """
        Heuristic: look for a monotonic timestamp-like column and estimate sampling rate.
        Returns detected sampling rate (Hz) or None if detection failed.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            col_vals = df[col].values
            if len(col_vals) < 3:
                continue
            diffs = np.diff(col_vals.astype(float))
            # Check monotonic increasing and mostly positive diffs
            positive_frac = np.mean(diffs > 0)
            if positive_frac < 0.9:
                continue
            median_diff = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else None
            if median_diff is None or median_diff <= 0:
                continue
            # If median_diff < 1 => units likely seconds (or fractions), compute sr = 1/median_diff
            try:
                if median_diff < 1.0:
                    sr = int(round(1.0 / median_diff))
                    if 1 <= sr <= 5000:
                        print(f"ℹ️ Detected sampling-rate candidate from column '{col}': median interval {median_diff:.6f} -> {sr} Hz")
                        return sr
                # If median_diff is large (e.g., in ms), try convert from ms to seconds
                if 1.0 <= median_diff < 10000:
                    # hypothesize milliseconds
                    sr = int(round(1000.0 / median_diff))
                    if 1 <= sr <= 5000:
                        print(f"ℹ️ Detected sampling-rate candidate from column '{col}' (assuming ms): median interval {median_diff:.6f} ms -> {sr} Hz")
                        return sr
            except Exception:
                continue
        print("ℹ️ Could not auto-detect sampling rate heuristically; using default/previous sampling_rate =", self.sampling_rate)
        return None
    
    def drop_non_eeg_columns(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Heuristics to drop columns that are likely non-EEG:
        - constant or near-constant columns (zero variance)
        - columns with mostly zeros
        - timestamp-like columns (monotonic increasing with low variance in diffs)
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        drop_cols = set()
        for col in numeric_cols:
            arr = df[col].values.astype(float)
            # zero or near-zero variance
            if np.nanstd(arr) == 0 or np.nanstd(arr) < 1e-6:
                drop_cols.add(col)
                if verbose:
                    print(f"   ↳ Dropping '{col}' (zero or near-zero variance)")
                continue
            # mostly zeros
            zero_frac = np.mean(arr == 0)
            if zero_frac > 0.98:
                drop_cols.add(col)
                if verbose:
                    print(f"   ↳ Dropping '{col}' (mostly zeros: {zero_frac:.2%})")
                continue
            # monotonic increasing timestamp-like (detect via diffs)
            diffs = np.diff(arr)
            positive_frac = np.mean(diffs >= 0) if len(diffs) > 0 else 0
            if positive_frac > 0.98:
                # if differences are fairly consistent (small std relative to mean), treat as timestamp
                mean_diff = np.mean(np.abs(diffs)) if len(diffs) > 0 else 0
                std_diff = np.std(diffs) if len(diffs) > 0 else 0
                if mean_diff > 0 and (std_diff / (abs(mean_diff) + 1e-12) < 0.5):
                    drop_cols.add(col)
                    if verbose:
                        print(f"   ↳ Dropping '{col}' (timestamp-like monotonic column)")
                    continue
            # huge integer ranges that could be IDs/labels (optional rule)
            if np.all(np.floor(arr) == arr):
                rng = np.max(arr) - np.min(arr)
                if rng > 1e6:
                    drop_cols.add(col)
                    if verbose:
                        print(f"   ↳ Dropping '{col}' (likely ID/marker with huge integer range {rng})")
                    continue
        
        cleaned_cols = [c for c in df.columns if c not in drop_cols]
        if len(cleaned_cols) == 0:
            print("⚠️ After dropping non-EEG columns, no numeric columns remain. Keeping at least the first numeric column.")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                cleaned_cols = [numeric_cols[0]]
        
        cleaned_df = df.loc[:, cleaned_cols].copy()
        return cleaned_df
    
    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        if low <= 0:
            low = 1e-6
        if high >= 0.999:
            high = 0.999
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def apply_bandpass_df(self, df: pd.DataFrame, lowcut=None, highcut=None, fs: Optional[float] = None) -> pd.DataFrame:
        """Apply zero-phase bandpass filtering to all numeric columns (returns filtered df)."""
        if lowcut is None:
            lowcut = self.lowcut
        if highcut is None:
            highcut = self.highcut
        if fs is None:
            fs = self.sampling_rate
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            return df
        # If sampling rate too low/unreliable, skip filtering
        if fs is None or fs <= 0:
            print("⚠️ Invalid sampling rate for filtering; skipping bandpass filter.")
            return df
        try:
            b, a = self.butter_bandpass(lowcut, highcut, fs, order=4)
            filtered = df.copy()
            for col in numeric_cols:
                col_data = df[col].values.astype(float)
                if len(col_data) < 3:
                    continue
                try:
                    # filtfilt can fail on small arrays - only apply if length sufficient
                    if len(col_data) < (3 * (max(len(a), len(b)))):
                        # too short for reliable filtfilt - skip filtering
                        continue
                    filtered_col = filtfilt(b, a, col_data, padlen=3*(max(len(a), len(b))-1))
                    filtered[col] = filtered_col
                except Exception:
                    # fallback to no filtering for that channel
                    continue
            return filtered
        except Exception as e:
            print(f"⚠️ Bandpass filter failed: {e} — proceeding without filtering.")
            return df
    
    def extract_features(self, eeg_signal: np.ndarray) -> np.ndarray:
        """Extract features from EEG signal."""
        features = []
        features.append(np.mean(eeg_signal))
        features.append(np.std(eeg_signal))
        features.append(np.max(eeg_signal))
        features.append(np.min(eeg_signal))
        fft = np.fft.fft(eeg_signal)
        fft_magnitude = np.abs(fft[:len(fft)//2])
        def safe_mean(slice_):
            if len(slice_) == 0:
                return 0.0
            return float(np.mean(slice_))
        features.append(safe_mean(fft_magnitude[:4]))    # Delta
        features.append(safe_mean(fft_magnitude[4:8]))   # Theta
        features.append(safe_mean(fft_magnitude[8:13]))  # Alpha
        features.append(safe_mean(fft_magnitude[13:30])) # Beta
        features.append(safe_mean(fft_magnitude[30:50])) # Gamma
        return np.array(features)
    
    def preprocess_dataset(self, df: pd.DataFrame, eeg_columns: List[str] = None, save_path: str = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process entire dataset: drop non-EEG cols, detect sampling rate, bandpass filter, extract features.
        Returns: (features_normalized, labels)
        Also saves features to disk at save_path (and CSV alongside .npy)
        """
        if save_path is None:
            save_path = str(PROCESSED_DATA_DIR / "processed_eeg_features.npy")
            
        if df is None or df.shape[0] == 0:
            raise ValueError("Empty or invalid dataframe provided to preprocess_dataset")
        
        # Detect sampling rate heuristic and update if found
        detected_sr = self.detect_sampling_rate(df)
        if detected_sr:
            self.sampling_rate = detected_sr
        
        # Drop non-EEG columns
        cleaned_df = self.drop_non_eeg_columns(df, verbose=True)
        
        # Identify EEG columns to use
        if eeg_columns is None:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
            non_eeg_cols = ['id', 'subject', 'trial', 'session', 'label', 'class', 'target', 'participant', 'sample']
            eeg_columns = [col for col in numeric_cols if col.lower() not in non_eeg_cols]
            if len(eeg_columns) == 0:
                eeg_columns = numeric_cols[:min(10, len(numeric_cols))]
        
        print(f"🔍 Using columns for EEG features: {eeg_columns[:10]}...")
        
        # Apply bandpass filter to numeric EEG columns
        filtered_df = self.apply_bandpass_df(cleaned_df[eeg_columns], fs=self.sampling_rate)
        
        features_list = []
        valid_rows = 0
        sample_size = min(100, len(filtered_df))
        step_size = max(1, len(filtered_df) // sample_size)
        
        for idx in range(0, len(filtered_df), step_size):
            if len(features_list) >= 100:
                break
            try:
                row_data = filtered_df.iloc[idx].values.astype(float)
                if len(row_data) > 0 and not np.any(np.isnan(row_data)) and np.all(np.isfinite(row_data)):
                    if len(row_data) > 20:
                        row_data = row_data[:20]
                    features = self.extract_features(row_data)
                    features_list.append(features)
                    valid_rows += 1
            except Exception as e:
                print(f"⚠️ Skipping row {idx} due to error: {e}")
                continue
        
        if len(features_list) == 0:
            print("⚠️ Row-based processing failed, trying column-based processing for time series...")
            for col_idx, col in enumerate(eeg_columns[:5]):
                if len(features_list) >= 100:
                    break
                try:
                    signal_series = filtered_df[col].dropna()
                    signal = signal_series.values
                    if len(signal) > 10:
                        chunk_size = min(1000, len(signal))
                        for start in range(0, len(signal), chunk_size):
                            if len(features_list) >= 100:
                                break
                            end = min(start + chunk_size, len(signal))
                            chunk = signal[start:end]
                            if len(chunk) > 10:
                                features = self.extract_features(chunk)
                                features_list.append(features)
                                valid_rows += 1
                except Exception as e:
                    print(f"⚠️ Skipping column {col} due to error: {e}")
                    continue
        
        if len(features_list) == 0:
            raise ValueError("No valid EEG data found in the dataset")
        
        features_array = np.array(features_list)
        features_normalized = self.scaler.fit_transform(features_array)
        
        # Save to disk
        import os
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        try:
            np.save(save_path, features_normalized)
            # write CSV copy for inspection
            csv_path = save_path.replace('.npy', '.csv')
            pd.DataFrame(features_normalized).to_csv(csv_path, index=False)
            print(f"✅ Saved processed features to {save_path} and {csv_path}")
        except Exception as e:
            print(f"⚠️ Could not save features to disk: {e}")
        
        print(f"✅ Extracted features from {valid_rows} samples: {features_normalized.shape}")
        return features_normalized, None