import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from src.models.generator import create_synthetic_eeg


def create_gradio_interface(generator):
    """Create the Gradio UI interface"""
    
    def generate_neuro_art(brain_activity: str, prompt: str, steps: int, guidance: float):
        """
        Gradio interface function
        """
        try:
            # Generate synthetic EEG based on activity level
            eeg_signal = create_synthetic_eeg(brain_activity)
            
            # Generate art
            image = generator.generate_art(
                eeg_signal,
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance
            )
            
            # Create EEG visualization
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(eeg_signal[:256], color='#00ff41', linewidth=1)
            ax.set_title('EEG Signal (1 second)', fontsize=14, color='white')
            ax.set_xlabel('Samples', color='white')
            ax.set_ylabel('Amplitude', color='white')
            ax.set_facecolor('#0a0e27')
            fig.patch.set_facecolor('#0a0e27')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.2, color='white')
            plt.tight_layout()
            
            return image, fig, "✅ Art generated successfully!"
        except Exception as e:
            return None, None, f"❌ Error: {str(e)}"
    
    # Custom CSS for professional look
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .gr-button-primary {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%) !important;
        border: none !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="NeuroCanvas - AI Art from Brainwaves") as demo:
        gr.Markdown("""
        # 🧠 NeuroCanvas: AI Art from Brainwaves
        ### Multi-Modal AI System | EEG Signals + Text → Unique Art Generation
        This application combines **brain activity patterns** with **text prompts** to generate unique AI artwork
        that reflects your mental state and imagination.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Input Controls")
                brain_activity = gr.Radio(
                    choices=["High Activity (Alert/Focused)",
                            "Medium Activity (Relaxed)",
                            "Low Activity (Drowsy/Meditative)"],
                    label="Brain Activity Level",
                    value="Medium Activity (Relaxed)",
                    info="Simulates different EEG patterns"
                )
                prompt_input = gr.Textbox(
                    label="Art Prompt",
                    placeholder="Enter your creative vision...",
                    value="a mystical brain floating in cosmic space",
                    lines=3
                )
                with gr.Accordion("Advanced Settings", open=False):
                    steps_slider = gr.Slider(
                        minimum=10, maximum=50, value=30, step=5,
                        label="Inference Steps (Higher = Better Quality)"
                    )
                    guidance_slider = gr.Slider(
                        minimum=1, maximum=15, value=7.5, step=0.5,
                        label="Guidance Scale (Higher = More Prompt Adherence)"
                    )
                generate_btn = gr.Button("🎨 Generate NeuroArt", variant="primary", size="lg")
                status_text = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=1):
                gr.Markdown("### 🖼️ Generated Artwork")
                output_image = gr.Image(label="AI Generated Art", type="pil")
                gr.Markdown("### 📊 EEG Signal Visualization")
                eeg_plot = gr.Plot(label="Brain Activity Pattern")
        
        # Examples
        gr.Markdown("### 💡 Example Prompts")
        gr.Examples(
            examples=[
                ["High Activity (Alert/Focused)", "neural networks glowing with electricity", 30, 7.5],
                ["Medium Activity (Relaxed)", "peaceful zen garden in the mind", 30, 7.5],
                ["Low Activity (Drowsy/Meditative)", "dreamlike clouds of consciousness", 30, 7.5],
                ["High Activity (Alert/Focused)", "cyberpunk brain interface with neon lights", 35, 8.0],
                ["Medium Activity (Relaxed)", "abstract thoughts flowing like water", 30, 7.0],
            ],
            inputs=[brain_activity, prompt_input, steps_slider, guidance_slider],
            outputs=[output_image, eeg_plot, status_text],
            fn=generate_neuro_art,
            cache_examples=False
        )
        
        # Event handlers
        generate_btn.click(
            fn=generate_neuro_art,
            inputs=[brain_activity, prompt_input, steps_slider, guidance_slider],
            outputs=[output_image, eeg_plot, status_text]
        )
        
        gr.Markdown("""
        ---
        ### 📚 About This Project
        **NeuroCanvas** is a multi-modal AI system that demonstrates:
        - ✅ EEG signal processing and feature extraction
        - ✅ Multi-modal fusion with neural networks
        - ✅ Integration with Stable Diffusion for image generation
        - ✅ Real-time interactive web interface
        **Tech Stack:** PyTorch, Transformers, Diffusers, CLIP, Gradio
        **Created by:** [Your Name] | AI/ML Engineer
        """)
    
    return demo