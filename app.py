import torch
import numpy as np
from config.settings import UI_SHARE, UI_DEBUG, FUSION_MODEL_PATH
from src.data.loader import DatasetLoader
from src.data.processor import EEGPreprocessor
from src.models.fusion import EEGTextFusionModel, FusionModelTrainer
from src.models.generator import NeuroCanvasGenerator
from src.ui.interface import create_gradio_interface


def main():
    """Main application entry point"""
    print("🚀 NeuroCanvas: AI Art from Brainwaves")
    print("=" * 70)
    print("Multi-Modal AI System | EEG + Text → Art Generation")
    print("=" * 70)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load or create EEG data
    print("\n🔍 Loading EEG data...")
    loader = DatasetLoader()
    df, data_path = loader.get_eeg_data()
    
    if df is not None:
        # Process real EEG data
        processor = EEGPreprocessor()
        eeg_features, _ = processor.preprocess_dataset(df)
        print(f"✅ Processed real EEG data: {eeg_features.shape}")
    else:
        # Use synthetic data
        print("⚠️ Using synthetic EEG data for demo")
        eeg_features = np.random.randn(100, 9)  # 100 samples, 9 features
        print(f"✅ Created synthetic EEG features: {eeg_features.shape}")
    
    # Initialize and train fusion model
    print("\n🧠 Initializing Multi-Modal Fusion Model...")
    fusion_model = EEGTextFusionModel().to(device)
    print(f"✅ Model initialized with {sum(p.numel() for p in fusion_model.parameters())} parameters")
    
    # Train the model
    trainer = FusionModelTrainer(fusion_model, device)
    
    # Check if model exists, if not train it
    if not FUSION_MODEL_PATH.exists():
        print("🎯 Training new model...")
        trained_model = trainer.train(eeg_features, epochs=10, lr=0.001)
        trainer.save_model()
    else:
        print("🎯 Loading existing model...")
        trained_model = trainer.load_model()
    
    # Initialize generator
    print("\n🎨 Initializing NeuroCanvas Generator...")
    generator = NeuroCanvasGenerator(trained_model, device)
    print("✅ NeuroCanvas Generator initialized!")
    
    # Create and launch UI
    print("\n🚀 Building Professional UI...")
    demo = create_gradio_interface(generator)
    
    print("\n" + "="*70)
    print("🎉 NeuroCanvas is Ready!")
    print("="*70)
    print("\n📝 INSTRUCTIONS:")
    print("   1. Select brain activity level")
    print("   2. Enter your creative prompt")
    print("   3. Click 'Generate NeuroArt'")
    print("   4. Watch as AI transforms your thoughts into art!")
    print("\n" + "="*70)
    
    # Launch the application
    demo.launch(
        share=UI_SHARE,
        debug=UI_DEBUG,
        server_name="127.0.0.1",
        server_port=7860
    )


if __name__ == "__main__":
    main()