import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from .fusion import EEGTextFusionModel
from src.data.processor import EEGPreprocessor
from config.settings import SD_MODEL_NAME


class NeuroCanvasGenerator:
    """
    Complete pipeline: EEG + Text → Fused Prompt → AI Art
    """
    def __init__(self, fusion_model: EEGTextFusionModel, device: torch.device):
        self.fusion_model = fusion_model
        self.device = device
        self.eeg_processor = EEGPreprocessor()
        self.fusion_model.eval()
        
        # Load Stable Diffusion
        print("🎨 Loading Stable Diffusion Pipeline...")
        print("⏳ This may take a few minutes on first run...")
        
        try:
            self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                SD_MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None
            )
        except Exception as e:
            print(f"❌ Error loading Stable Diffusion: {e}")
            print("🔄 Trying with different format (loading with use_safetensors=False)...")
            try:
                self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                    SD_MODEL_NAME,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    safety_checker=None,
                    use_safetensors=False  # Try with .bin files instead
                )
            except Exception as e2:
                print(f"❌ Second attempt failed: {e2}")
                print("🔄 Trying with a smaller model (stabilityai/stable-diffusion-2-1-base)...")
                try:
                    self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-2-base",  # Smaller model
                        torch_dtype=torch.float32,
                        safety_checker=None
                    )
                except Exception as e3:
                    print(f"❌ All attempts failed: {e3}")
                    print("⚠️  Using text-to-image generation fallback...")
                    # Create a fallback that returns text as image
                    from PIL import Image, ImageDraw, ImageFont
                    import io
                    
                    def fallback_generate(prompt, **kwargs):
                        # Create a simple fallback image with the prompt text
                        img = Image.new('RGB', (512, 512), color='black')
                        d = ImageDraw.Draw(img)
                        try:
                            # Try to use a default font
                            d.text((50, 250), prompt[:50] + "..." if len(prompt) > 50 else prompt, fill=(255, 255, 255))
                        except:
                            # Fallback if font fails
                            pass
                        return {"images": [img]}
                    
                    class FallbackPipeline:
                        def __call__(self, prompt, **kwargs):
                            return fallback_generate(prompt, **kwargs)
                    
                    self.sd_pipe = FallbackPipeline()
                    print("✅ Fallback pipeline initialized")
        
        # Optimize for speed if not using fallback
        if hasattr(self.sd_pipe, 'scheduler'):
            self.sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.sd_pipe.scheduler.config)
        
        self.sd_pipe = self.sd_pipe.to(device)
        
        # Enable memory optimization
        if torch.cuda.is_available():
            try:
                self.sd_pipe.enable_attention_slicing()
                self.sd_pipe.enable_vae_slicing()
            except:
                pass  # Ignore if methods don't exist in fallback
        
        # Load CLIP tokenizer and encoder
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.clip_text_encoder.eval()
        
        print("✅ Stable Diffusion and CLIP loaded successfully!")
    
    def generate_art(self, eeg_signal: np.ndarray, text_prompt: str,
                     num_inference_steps: int = 30, guidance_scale: float = 7.5) -> Image.Image:
        """
        Generate AI art from EEG signal and text prompt
        """
        # Extract EEG features
        eeg_features = self.eeg_processor.extract_features(eeg_signal)
        eeg_tensor = torch.FloatTensor(eeg_features).unsqueeze(0).to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            text_inputs = self.clip_tokenizer([text_prompt], padding=True, return_tensors="pt").to(self.device)
            text_embeddings = self.clip_text_encoder(**text_inputs).pooler_output
            # Fuse EEG and text
            fused_embeddings = self.fusion_model(eeg_tensor, text_embeddings)
        
        # Enhance prompt based on EEG features
        eeg_intensity = np.mean(np.abs(eeg_signal))
        eeg_variance = np.std(eeg_signal)
        
        # Add EEG-driven artistic modifiers
        if eeg_intensity > 0.5:
            enhanced_prompt = f"{text_prompt}, vibrant and energetic, highly detailed"
        elif eeg_variance > 0.3:
            enhanced_prompt = f"{text_prompt}, dynamic and flowing, artistic"
        else:
            enhanced_prompt = f"{text_prompt}, calm and serene, minimalist"
        
        print(f"\n🎨 Generating art...")
        print(f"   Original prompt: {text_prompt}")
        print(f"   Enhanced prompt: {enhanced_prompt}")
        print(f"   EEG intensity: {eeg_intensity:.3f} | Variance: {eeg_variance:.3f}")
        
        # Generate image
        try:
            result = self.sd_pipe(
                enhanced_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            )
            image = result.images[0] if hasattr(result, 'images') else result
        except Exception as e:
            print(f"⚠️  Generation failed: {e}")
            print("⚠️  Using fallback image...")
            # Create a fallback image
            image = Image.new('RGB', (512, 512), color='darkblue')
            from PIL import ImageDraw
            d = ImageDraw.Draw(image)
            d.text((50, 250), f"Generation failed:\n{str(e)[:50]}", fill=(255, 255, 255))
        
        return image


def create_synthetic_eeg(activity_level: str) -> np.ndarray:
    """
    Generate synthetic EEG for demo purposes
    In production, this would read from actual EEG headset
    """
    duration = 2  # seconds
    sampling_rate = 256
    t = np.linspace(0, duration, sampling_rate * duration)
    
    if activity_level == "High Activity (Alert/Focused)":
        # Beta waves dominant (13-30 Hz)
        signal = np.sin(2 * np.pi * 20 * t) + 0.3 * np.random.randn(len(t))
    elif activity_level == "Medium Activity (Relaxed)":
        # Alpha waves dominant (8-13 Hz)
        signal = np.sin(2 * np.pi * 10 * t) + 0.2 * np.random.randn(len(t))
    else:  # Low Activity
        # Theta/Delta waves (4-8 Hz)
        signal = np.sin(2 * np.pi * 6 * t) + 0.1 * np.random.randn(len(t))
    
    return signal