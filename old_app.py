import gradio as gr
import os
import shutil
from generate import generate_music
from visualize import plot_piano_roll
from magenta_integration import generate_with_magenta

def ui_generate(model_choice, temperature, generate_length):
    """
    Wrapper for generation logic to be used by Gradio UI.
    """
    output_filename = f"generated_temp_{temperature}_len_{generate_length}.mid"
    output_image = None
    
    try:
        if model_choice == "Magenta (Pre-trained)":
            output_path = generate_with_magenta(
                output_filename=output_filename,
                num_steps=int(generate_length),
                temperature=float(temperature)
            )
            if not output_path:
                return None, None, "Error: Magenta generation failed."
        elif model_choice == "Music Transformer" or model_choice == "GAN (Experimental)":
            # These require specific training which might not be complete yet.
            # We'll use the LSTM generator as a fallback or return an error if it's strictly needed.
            # For demonstration, we'll return a message.
            return None, None, f"{model_choice} selected, but weights are not trained. Please use LSTM or Magenta."
        else:
            # Default to LSTM
            model_path = "data/models/final_model.keras"
            notes_path = "data/processed/notes.pkl"
            
            if not os.path.exists(model_path):
                return None, None, "Error: Trained LSTM model not found."
                
            if not os.path.exists(notes_path):
                return None, None, "Error: Processed notes not found."
                
            output_path = generate_music(
                model_path=model_path,
                notes_path=notes_path,
                temperature=float(temperature),
                generate_length=int(generate_length),
                output_filename=output_filename
            )
            
        if output_path and os.path.exists(output_path):
            # Generate Piano Roll
            image_path = plot_piano_roll(output_path)
            return output_path, image_path, f"Music generated successfully using {model_choice}!"
        else:
            return None, None, "Error during generation process."
            
    except Exception as e:
        return None, None, f"An error occurred: {str(e)}"

# Define the Gradio Interface
with gr.Blocks(title="Advanced Music Generation AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎵 Advanced Music Generation AI")
    gr.Markdown("Generate new MIDI music sequences using advanced Deep Learning models (LSTM, Transformer, GAN) or Google's Magenta.")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=["LSTM (Custom)", "Music Transformer", "GAN (Experimental)", "Magenta (Pre-trained)"],
                value="LSTM (Custom)",
                label="Model Selection",
                info="Select the architecture for generation."
            )
            temp_slider = gr.Slider(
                minimum=0.1, 
                maximum=2.0, 
                value=1.0, 
                step=0.1, 
                label="Temperature", 
                info="Higher values = more creative/random. Lower values = more predictable."
            )
            length_slider = gr.Slider(
                minimum=50, 
                maximum=1000, 
                value=500, 
                step=50, 
                label="Generation Length", 
                info="Number of notes/chords to generate."
            )
            generate_btn = gr.Button("Generate Music", variant="primary")
            
        with gr.Column(scale=1):
            output_status = gr.Textbox(label="Status", interactive=False)
            output_file = gr.File(label="Download Generated MIDI", interactive=False)
            output_plot = gr.Image(label="Piano-Roll Visualization")
            
    # Link button to function
    generate_btn.click(
        fn=ui_generate,
        inputs=[model_dropdown, temp_slider, length_slider],
        outputs=[output_file, output_plot, output_status]
    )

if __name__ == "__main__":
    # Launch the web UI
    demo.launch(share=False)
