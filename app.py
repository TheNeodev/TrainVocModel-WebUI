import gradio as gr
import os
import subprocess
import json
from random import shuffle
BASE_DIR = os.getcwd()
# Helper Functions
def data_processing(model_name, dataset_folder, sample_rate, is_half, new_preprocess, percentage):
    try:
        sr = int(sample_rate.rstrip("k")) * 1000
        preprocess_file = 'preprocess_custom.py' if new_preprocess else 'preprocess.py'

        # Create logs directory for the model
        os.makedirs(f'./logs/{model_name}', exist_ok=True)

        # Run the preprocessing script
        subprocess.run(
            [
                "python", f"infer/modules/train/{preprocess_file}", dataset_folder, 
                str(sr), "2", f"./logs/{model_name}", "False", str(percentage)
            ], check=True
        )

        # Check logs for successful preprocessing
        with open(f'./logs/{model_name}/preprocess.log', 'r') as log:
            if 'end preprocess' not in log.read():
                raise Exception("Data preprocessing failed. Check dataset folder.")

        return f"Data processing completed for model: {model_name}"
    except Exception as e:
        return str(e)


def model_training(
    model_name, sample_rate, epochs, save_epoch, pretrain, custom_pretrained, 
    d_pretrained_link, g_pretrained_link, batch_size, fp16_run, tensorboard
):
    try:
        pretrain_outpath = "/content/pretrained_models"
        os.makedirs(pretrain_outpath, exist_ok=True)

        # Handle pretrained model selection
        if custom_pretrained and d_pretrained_link and g_pretrained_link:
            subprocess.run(["aria2c", d_pretrained_link, "-d", pretrain_outpath], check=True)
            subprocess.run(["aria2c", g_pretrained_link, "-d", pretrain_outpath], check=True)
            G_file = os.path.join(pretrain_outpath, os.path.basename(g_pretrained_link))
            D_file = os.path.join(pretrain_outpath, os.path.basename(d_pretrained_link))
        else:
            # Use predefined pretrained models
            G_file = f"{pretrain_outpath}"  # Define based on your mapping logic
            D_file = f"{pretrain_outpath}"  # Define based on your mapping logic

        # Write training configuration
        exp_dir = f"./logs/{model_name}"
        os.makedirs(exp_dir, exist_ok=True)
        config_path = f"configs/v2/{sample_rate}.json"
        config_save_path = os.path.join(exp_dir, "config.json")

        with open(config_save_path, "w") as f:
            with open(config_path, "r") as config_file:
                config_data = json.load(config_file)
                config_data["train"]["fp16_run"] = fp16_run
                json.dump(config_data, f, indent=4)

        # Run the training process
        training_command = [
            "python", "infer/modules/train/train.py",
            "-e", model_name, "-sr", sample_rate, "-f0", "1", "-bs", str(batch_size),
            "-te", epochs, "-se", save_epoch,
            "-pg", G_file, "-pd", D_file
        ]

        subprocess.run(training_command, check=True)
        return "Model training started successfully!"
    except Exception as e:
        return str(e)


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("### Model Training WebUI")
    
    with gr.Tab("Data Processing"):
        model_name = gr.Textbox(label="Model Name", placeholder="Enter model name")
        dataset_folder = gr.Textbox(label="Dataset Folder", value="/content/dataset")
        sample_rate = gr.Radio(["32k", "40k", "48k"], value="40k", label="Sample Rate")
        is_half = gr.Checkbox(value=True, label="Memory Saving")
        new_preprocess = gr.Checkbox(value=False, label="Use New Preprocess")
        percentage = gr.Slider(0.1, 10.0, value=3.7, label="Percentage")
        
        process_button = gr.Button("Start Data Processing")
        process_output = gr.Textbox(label="Output")

        process_button.click(
            data_processing, 
            inputs=[model_name, dataset_folder, sample_rate, is_half, new_preprocess, percentage],
            outputs=process_output
        )

    with gr.Tab("Model Training"):
        epochs = gr.Number(value=500, label="Epochs")
        save_epoch = gr.Number(value=50, label="Save Epoch Frequency")
        pretrain = gr.Radio(
            [
                "* Default —> (Sampling — ALL)", "* Snowie —> (Sampling — 40k)", 
                "* Ov2Super —> (Sampling — 40k)"
            ], value="* Default —> (Sampling — ALL)", label="Pretrained Model"
        )
        custom_pretrained = gr.Checkbox(value=False, label="Use Custom Pretrained")
        d_pretrained_link = gr.Textbox(label="D Pretrained Link")
        g_pretrained_link = gr.Textbox(label="G Pretrained Link")
        batch_size = gr.Slider(4, 32, step=4, value=8, label="Batch Size")
        fp16_run = gr.Checkbox(value=True, label="Enable FP16")
        tensorboard = gr.Checkbox(value=False, label="Enable TensorBoard")

        train_button = gr.Button("Start Model Training")
        train_output = gr.Textbox(label="Output")

        train_button.click(
            model_training, 
            inputs=[
                model_name, sample_rate, epochs, save_epoch, pretrain, 
                custom_pretrained, d_pretrained_link, g_pretrained_link, 
                batch_size, fp16_run, tensorboard
            ], outputs=train_output
        )

demo.launch()
