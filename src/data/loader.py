import os
import shutil
import subprocess
import json
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from scipy.io import loadmat


class DatasetLoader:
    """
    Handles dataset loading from multiple sources:
    - Kaggle datasets (with required authentication)
    - MNE open datasets
    - Local files
    - Synthetic data generation
    """
    
    def __init__(self):
        self.kaggle_available = self._check_kaggle_availability()
        self.kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
        
    def _check_kaggle_availability(self) -> bool:
        """Check if Kaggle CLI is available."""
        try:
            result = subprocess.run(['kaggle', '-v'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def authenticate_kaggle(self) -> bool:
        """
        Handle Kaggle authentication - mimics the Google Colab flow
        """
        print("\n📁 Kaggle & Open EEG Dataset Setup (will ask for kaggle.json)")
        print("-" * 70)
        print("Behavior:")
        print("  • This will ask you to provide kaggle.json")
        print("  • After that it will try Kaggle downloads, then MNE open datasets, then synthetic fallback.")
        print("-" * 70)
        
        # Check if kaggle.json already exists
        if self.kaggle_json_path.exists():
            print(f"✅ Found existing kaggle.json at: {self.kaggle_json_path}")
            try:
                with open(self.kaggle_json_path, 'r') as f:
                    kaggle_config = json.load(f)
                print(f"ℹ️ kaggle.json keys: {', '.join(kaggle_config.keys())}")
                return True
            except Exception as e:
                print(f"⚠️ Error reading existing kaggle.json: {e}")
        
        # Force prompt for kaggle.json
        while True:
            print("\n📥 Please provide your kaggle.json file.")
            print("    If you DO NOT want to provide kaggle credentials, type 'skip' when prompted below.")
            
            try:
                choice = input("Enter 'upload' to provide file path, 'skip' to proceed without Kaggle, or 'help' for instructions: ").strip().lower()
            except Exception:
                choice = 'upload'
            
            if choice == 'skip':
                print("ℹ️ Proceeding WITHOUT Kaggle credentials as requested.")
                return False
            elif choice == 'help':
                print("\n📝 Instructions:")
                print("1. Go to Kaggle.com → Account → API → Create New API Token")
                print("2. Download kaggle.json file")
                print("3. Enter the path to your kaggle.json file when prompted")
                print("4. Or copy it to ~/.kaggle/kaggle.json manually")
                continue
            elif choice == 'upload':
                # Get file path from user
                file_path = input("Enter the full path to your kaggle.json file: ").strip()
                
                if not os.path.exists(file_path):
                    print("❌ File does not exist!")
                    retry = input("Retry? (Y/n): ").strip().lower()
                    if retry == '' or retry.startswith('y'):
                        continue
                    else:
                        print("ℹ️ Proceeding without Kaggle credentials.")
                        return False
                
                # Validate JSON format
                try:
                    with open(file_path, 'r') as f:
                        kaggle_config = json.load(f)
                    if 'username' not in kaggle_config or 'key' not in kaggle_config:
                        raise ValueError("Missing username or key in kaggle.json")
                except Exception as e:
                    print(f"❌ Invalid kaggle.json format: {e}")
                    retry = input("Retry? (Y/n): ").strip().lower()
                    if retry == '' or retry.startswith('y'):
                        continue
                    else:
                        print("ℹ️ Proceeding without Kaggle credentials.")
                        return False
                
                # Save to correct location
                try:
                    self.kaggle_json_path.parent.mkdir(exist_ok=True)
                    shutil.copy2(file_path, self.kaggle_json_path)
                    os.chmod(self.kaggle_json_path, 0o600)  # Secure permissions
                    print(f"✅ Saved kaggle.json to: {self.kaggle_json_path}")
                    print(f"ℹ️ kaggle.json keys: {', '.join(kaggle_config.keys())}")
                    return True
                except Exception as e:
                    print(f"❌ Failed to save kaggle.json: {e}")
                    retry = input("Retry? (Y/n): ").strip().lower()
                    if retry == '' or retry.startswith('y'):
                        continue
                    else:
                        print("ℹ️ Proceeding without Kaggle credentials.")
                        return False
            else:
                print("Invalid choice. Please enter 'upload', 'skip', or 'help'")
    
    def _run_cmd(self, cmd: str) -> tuple[bool, list]:
        """Run shell command and return (success, output_lines)"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return True, result.stdout.splitlines()
            else:
                return False, (result.stderr if result.stderr else result.stdout).splitlines()
        except Exception as e:
            return False, [str(e)]
    
    def download_kaggle_datasets(self) -> List[str]:
        """Download EEG datasets from Kaggle."""
        if not self.kaggle_available:
            print("⚠️ Kaggle CLI not available - skipping Kaggle downloads")
            return []
        
        if not self.kaggle_json_path.exists():
            print("⚠️ No kaggle.json found - skipping Kaggle downloads")
            return []
        
        kaggle_datasets = [
            "inancigdem/eeg-data-for-mental-attention-state-detection",
            "birdoring/berkeley-eeg",
            "eegdata/eeg-brainwave-detection"
        ]
        
        found_files = []
        temp_dir = RAW_DATA_DIR / "temp_kaggle"
        temp_dir.mkdir(exist_ok=True)
        
        for dataset in kaggle_datasets:
            print(f"\n📥 Trying Kaggle dataset: {dataset}")
            success, _ = self._run_cmd(f"rm -rf {temp_dir}/*")
            
            ds_for_cmd = dataset.replace("kaggle/datasets/", "")
            success, out_lines = self._run_cmd(f"kaggle datasets download -d {ds_for_cmd} -p {temp_dir} --unzip")
            
            if success:
                # Scan for candidate files
                candidate_exts = ('.csv', '.mat', '.xlsx', '.xls', '.txt', '.dat', '.json', '.edf')
                found_dataset_files = []
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith(candidate_exts):
                            found_dataset_files.append(str(Path(root) / file))
                
                if found_dataset_files:
                    dest_dir = RAW_DATA_DIR / dataset.replace('/', '_')
                    dest_dir.mkdir(exist_ok=True)
                    for f in found_dataset_files:
                        try:
                            shutil.copy(f, dest_dir)
                            found_files.append(str(dest_dir / Path(f).name))
                        except:
                            pass
                    
                    print(f"✅ Kaggle dataset {dataset} downloaded and copied to {dest_dir}")
                    break
                else:
                    print("⚠️ Kaggle archive downloaded but no usable EEG file types found in it.")
            else:
                print("⚠️ Kaggle download failed or dataset not public/accessible; trying next.")
        
        # Clean up temp directory
        try:
            self._run_cmd(f"rm -rf {temp_dir}")
        except:
            pass
        
        return found_files
    
    def load_mne_datasets(self) -> List[str]:
        """Load open-source EEG datasets using MNE."""
        try:
            import mne
            found_files = []
            
            # Try eegbci dataset
            try:
                print("\n🌐 Attempting MNE eegbci dataset download...")
                files_list = mne.datasets.eegbci.load_data(
                    subject=1, runs=[3], 
                    path=RAW_DATA_DIR / "mne_eegbci",
                    update_path=True, verbose=False
                )
                found_files.extend([str(f) for f in files_list])
                print(f"✅ eegbci files downloaded: {len(files_list)} files")
            except Exception as e:
                print(f"⚠️ eegbci download failed: {e}")
            
            # Try sample dataset
            try:
                print("🌐 Attempting MNE sample dataset...")
                sample_path = mne.datasets.sample.data_path(
                    path=RAW_DATA_DIR / "mne_sample", verbose=False
                )
                if sample_path and os.path.exists(sample_path):
                    found_files.append(str(sample_path))
                    print(f"✅ MNE sample dataset available: {sample_path}")
            except Exception as e:
                print(f"⚠️ MNE sample download failed: {e}")
            
            return found_files
            
        except ImportError:
            print("⚠️ MNE not available - skipping MNE datasets")
            return []
    
    def discover_eeg_files(self, search_dir: Optional[Path] = None) -> List[str]:
        """Discover EEG files in directory."""
        if search_dir is None:
            search_dir = RAW_DATA_DIR
            
        if not search_dir.exists():
            return []
            
        candidate_exts = ('.csv', '.mat', '.xlsx', '.xls', '.txt', '.dat', '.json', '.edf')
        found_files = []
        
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.lower().endswith(candidate_exts):
                    found_files.append(str(Path(root) / file))
        
        return sorted(list(set(found_files)))
    
    def load_dataset(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load EEG data from various formats."""
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
                print(f"✅ Loaded CSV: {filepath} -> {df.shape}")
                return df
            elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                df = pd.read_excel(filepath)
                print(f"✅ Loaded Excel: {filepath} -> {df.shape}")
                return df
            elif filepath.endswith('.txt'):
                df = pd.read_csv(filepath, sep=None, engine='python')
                print(f"✅ Loaded TXT: {filepath} -> {df.shape}")
                return df
            elif filepath.endswith('.mat'):
                return self._load_mat_file(filepath)
            else:
                df = pd.read_csv(filepath, sep=None, engine='python')
                print(f"✅ Fallback: read file as CSV-like: {df.shape}")
                return df
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return None
    
    def _load_mat_file(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load MATLAB .mat files with robust unwrapping."""
        try:
            mat_data = loadmat(filepath, struct_as_record=False, squeeze_me=True)
            print(f"🔍 Available keys in .mat file: {list(mat_data.keys())}")
            
            candidate_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            main_data = None
            
            if 'o' in mat_data:
                main_data = mat_data['o']
                print("🎯 Using 'o' key for data (present).")
            elif candidate_keys:
                preferred = None
                for k in candidate_keys:
                    if k.lower() in ['o', 'data', 'eeg', 'signal', 'x', 'y', 's']:
                        preferred = k
                        break
                use_key = preferred if preferred is not None else candidate_keys[0]
                main_data = mat_data[use_key]
                print(f"🎯 Using key: {use_key}")
            else:
                raise ValueError("No non-internal keys found in .mat file")
            
            def unwrap(obj, depth=0):
                prefix = "  " * depth
                if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.number):
                    print(f"{prefix}➡ Found numeric ndarray with shape {obj.shape}")
                    return obj
                if isinstance(obj, np.ndarray) and obj.dtype == object:
                    if obj.size == 1:
                        elem = obj.flat[0]
                        print(f"{prefix}↳ Unwrapping object-array singleton -> {type(elem)}")
                        return unwrap(elem, depth+1)
                    else:
                        try:
                            arr = np.array(obj.tolist(), dtype=float)
                            print(f"{prefix}↳ Converted object-array to numeric ndarray with shape {arr.shape}")
                            return arr
                        except Exception:
                            for idx, item in enumerate(obj.flat):
                                try:
                                    candidate = unwrap(item, depth+1)
                                    if isinstance(candidate, np.ndarray) and np.issubdtype(candidate.dtype, np.number):
                                        print(f"{prefix}↳ Using element {idx} of object-array as numeric data")
                                        return candidate
                                except Exception:
                                    continue
                            raise ValueError(f"{prefix}No numeric content found inside object ndarray of shape {obj.shape}")
                
                if hasattr(obj, '__dict__') or hasattr(obj, '__slots__'):
                    try:
                        fields = [a for a in dir(obj) if not a.startswith('_')]
                    except Exception:
                        fields = []
                    for fld in fields:
                        try:
                            val = getattr(obj, fld)
                            if val is None:
                                continue
                            candidate = unwrap(val, depth+1)
                            if isinstance(candidate, np.ndarray) and np.issubdtype(candidate.dtype, np.number):
                                print(f"{prefix}↳ Extracted numeric ndarray from attribute '{fld}'")
                                return candidate
                        except Exception:
                            continue
                
                if hasattr(obj, 'dtype') and getattr(obj.dtype, 'names', None):
                    for name in obj.dtype.names:
                        try:
                            val = obj[name]
                            candidate = unwrap(val, depth+1)
                            if isinstance(candidate, np.ndarray) and np.issubdtype(candidate.dtype, np.number):
                                print(f"{prefix}↳ Extracted numeric ndarray from structured field '{name}'")
                                return candidate
                        except Exception:
                            continue
                    raise ValueError(f"{prefix}Structured array contains no numeric fields: {obj.dtype.names}")
                
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        try:
                            candidate = unwrap(v, depth+1)
                            if isinstance(candidate, np.ndarray) and np.issubdtype(candidate.dtype, np.number):
                                print(f"{prefix}↳ Extracted numeric ndarray from dict key '{k}'")
                                return candidate
                        except Exception:
                            continue
                    raise ValueError(f"{prefix}Dict contains no numeric arrays")
                
                if isinstance(obj, (list, tuple)):
                    for i, item in enumerate(obj):
                        try:
                            candidate = unwrap(item, depth+1)
                            if isinstance(candidate, np.ndarray) and np.issubdtype(candidate.dtype, np.number):
                                print(f"{prefix}↳ Extracted numeric ndarray from list/tuple index {i}")
                                return candidate
                        except Exception:
                            continue
                    raise ValueError(f"{prefix}List/tuple contains no numeric arrays")
                
                if np.isscalar(obj) and np.isfinite(obj):
                    arr = np.array([obj], dtype=float)
                    print(f"{prefix}➡ Scalar numeric converted to array: {arr.shape}")
                    return arr
                
                raise ValueError(f"{prefix}Could not find numeric ndarray inside object of type {type(obj)}")
            
            numeric = unwrap(main_data)
            
            # Identify sampling rate metadata
            try:
                if hasattr(main_data, 'fs'):
                    fs = getattr(main_data, 'fs')
                    if np.isscalar(fs) and fs > 0:
                        print(f"ℹ️ Detected sampling rate from 'fs': {int(fs)} Hz")
            except Exception:
                pass
            
            if numeric.ndim == 1:
                df = pd.DataFrame({'signal': numeric})
                print(f"✅ Converted 1D numeric array to DataFrame: {df.shape}")
                return df
            elif numeric.ndim == 2:
                rows, cols = numeric.shape
                if rows < cols:
                    df = pd.DataFrame(numeric.T)
                else:
                    df = pd.DataFrame(numeric)
                df.columns = [f'channel_{i}' for i in range(df.shape[1])]
                print(f"✅ Converted 2D numeric array to DataFrame: {df.shape}")
                return df
            elif numeric.ndim == 3:
                first_trial = numeric[0]
                df = pd.DataFrame(first_trial.T)
                df.columns = [f'channel_{i}' for i in range(df.shape[1])]
                print(f"✅ Converted 3D numeric array (first trial) to DataFrame: {df.shape}")
                return df
            else:
                flat = numeric.reshape(numeric.shape[0], -1)
                df = pd.DataFrame(flat)
                df.columns = [f'col_{i}' for i in range(df.shape[1])]
                print(f"✅ Flattened higher-dim numeric array to DataFrame: {df.shape}")
                return df
                
        except Exception as e:
            print(f"❌ Error loading .mat file {filepath}: {e}")
            return None
    
    def get_eeg_data(self) -> tuple[Optional[pd.DataFrame], str]:
        """Get EEG data from all available sources."""
        # Authenticate with Kaggle first
        kaggle_authenticated = self.authenticate_kaggle()
        
        print("\n🔍 Searching for EEG datasets...")
        
        # Discover existing files
        existing_files = self.discover_eeg_files()
        if existing_files:
            print(f"✅ Found {len(existing_files)} existing EEG files")
            # Use the largest file
            largest_file = max(existing_files, key=lambda x: os.path.getsize(x))
            print(f"🎯 Using largest file: {largest_file}")
            df = self.load_dataset(largest_file)
            return df, largest_file
        
        # Try Kaggle datasets if authenticated
        if kaggle_authenticated:
            kaggle_files = self.download_kaggle_datasets()
            if kaggle_files:
                df = self.load_dataset(kaggle_files[0])
                return df, kaggle_files[0]
        
        # Try MNE datasets
        mne_files = self.load_mne_datasets()
        if mne_files:
            # For MNE, we need to handle the data differently
            # Return synthetic data as fallback since MNE data is complex
            pass
        
        print("⚠️ No real EEG data found - will use synthetic data")
        return None, ""