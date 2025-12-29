import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from config.settings import FUSION_MODEL_PATH


class EEGTextFusionModel(nn.Module):
    """
    Neural network that fuses EEG features with text embeddings
    Architecture: EEG Encoder → Fusion Layer → Text Embedding Space
    """
    def __init__(self, eeg_feature_dim=9, text_embedding_dim=768, hidden_dim=512):
        super(EEGTextFusionModel, self).__init__()
        # EEG Encoder
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, text_embedding_dim)
        )
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(text_embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, text_embedding_dim)
        )
    
    def forward(self, eeg_features, text_embeddings):
        """
        Forward pass: Combine EEG and text features
        """
        eeg_encoded = self.eeg_encoder(eeg_features)
        # Concatenate EEG and text embeddings
        combined = torch.cat([eeg_encoded, text_embeddings], dim=1)
        # Fuse features
        fused = self.fusion(combined)
        return fused


class EEGTextDataset(Dataset):
    """Custom dataset for EEG-Text pairs"""
    def __init__(self, eeg_features, prompts):
        self.eeg_features = torch.FloatTensor(eeg_features)
        self.prompts = prompts
    
    def __len__(self):
        return len(self.eeg_features)
    
    def __getitem__(self, idx):
        return self.eeg_features[idx], self.prompts[idx]


class FusionModelTrainer:
    """Handles training of the EEG-Text fusion model"""
    
    def __init__(self, model: EEGTextFusionModel, device: torch.device):
        self.model = model
        self.device = device
        self.clip_tokenizer = None
        self.clip_text_encoder = None
        self._load_clip_models()
    
    def _load_clip_models(self):
        """Load CLIP models for text encoding"""
        print("📝 Loading CLIP text encoder...")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_text_encoder.eval()
        print("✅ CLIP models loaded")
    
    def train(self, eeg_features: np.ndarray, epochs: int = 10, lr: float = 0.001):
        """Train the fusion model"""
        # Create training prompts
        training_prompts = [
            "abstract neural patterns with vibrant colors",
            "cosmic brain waves flowing through space",
            "electric thoughts visualized as art",
            "meditation state captured in colors",
            "focused concentration as geometric patterns",
            "creative thinking represented as flowing shapes",
            "calm mind depicted as smooth gradients",
            "active brain shown as dynamic fractals",
            "dreamlike neural activity in watercolor style",
            "analytical thinking as structured forms"
        ] * (len(eeg_features) // 10 + 1)
        training_prompts = training_prompts[:len(eeg_features)]
        
        # Create dataset and dataloader
        train_dataset = EEGTextDataset(eeg_features, training_prompts)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        print("🎓 Training Multi-Modal Fusion Model...")
        print("=" * 70)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, (eeg_batch, prompts) in enumerate(train_loader):
                eeg_batch = eeg_batch.to(self.device)
                
                # Get text embeddings from CLIP
                with torch.no_grad():
                    text_inputs = self.clip_tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
                    text_embeddings = self.clip_text_encoder(**text_inputs).pooler_output
                
                # Forward pass
                optimizer.zero_grad()
                fused_embeddings = self.model(eeg_batch, text_embeddings)
                
                # Loss: Make fused embedding close to text embedding (contrastive learning)
                loss = criterion(fused_embeddings, text_embeddings)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")
        
        print("✅ Training completed!")
        return self.model
    
    def save_model(self, path: str = None):
        """Save the trained model"""
        if path is None:
            path = str(FUSION_MODEL_PATH)
        
        torch.save(self.model.state_dict(), path)
        print(f"✅ Model saved: {path}")
    
    def load_model(self, path: str = None):
        """Load a trained model"""
        if path is None:
            path = str(FUSION_MODEL_PATH)
        
        if not torch.load(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✅ Model loaded: {path}")
        return self.model