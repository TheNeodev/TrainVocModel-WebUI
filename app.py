import gradio as gr
import os
import re
import numpy as np
import subprocess
from sklearn.cluster import MiniBatchKMeans
from multiprocessing import cpu_count
import traceback
import faiss

# Helper Functions
def validate_model_name(name):
    if not re.match(r'^[\w_-]+$', name):
        return "Invalid model name: Contains invalid characters or spaces!"
    return None

def preprocess_dataset(model_name, dataset_folder, sample_rate, is_half, new_preprocess, percentage):
    if not os.listdir(dataset_folder):
        return "Dataset folder is empty!"

    preprocess_file = 'preprocess_custom.py' if new_preprocess else 'preprocess.py'

    if new_preprocess and not os.path.isfile(f'infer/modules/train/{preprocess_file}'):
        preprocess_file = 'preprocess.py'

    os.makedirs(f'./logs/{model_name}', exist_ok=True)
    with open(f'./logs/{model_name}/preprocess.log', 'w') as f:
        subprocess.run([
            "python", f"infer/modules/train/{preprocess_file}", dataset_folder, str(sample_rate), "2", f"./logs/{model_name}", "False", str(percentage)
        ])

    with open(f'./logs/{model_name}/preprocess.log', 'r') as f:
        log_content = f.read()
        if 'end preprocess' in log_content:
            return "Preprocessing complete!"
        else:
            return "Error during preprocessing. Check dataset folder and logs."

def train_index(model_name, version):
    try:
        exp_dir = f"logs/{model_name}"
        os.makedirs(exp_dir, exist_ok=True)
        feature_dir = f"{exp_dir}/3_feature256" if version == "v1" else f"{exp_dir}/3_feature768"

        if not os.path.exists(feature_dir):
            return "Feature extraction required before index training!"

        npys = [np.load(f"{feature_dir}/{name}") for name in os.listdir(feature_dir)]
        big_npy = np.concatenate(npys, axis=0)
        
        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]

        if big_npy.shape[0] > 200000:
            big_npy = MiniBatchKMeans(n_clusters=10000, verbose=True, batch_size=256 * cpu_count(), compute_labels=False, init="random").fit(big_npy).cluster_centers_

        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        index = faiss.index_factory(256 if version == "v1" else 768, f"IVF{n_ivf},Flat")
        index.train(big_npy)
        index.add(big_npy)

        faiss.write_index(index, f"{exp_dir}/trained_IVF{n_ivf}_Flat.index")
        return "Index training completed successfully!"
    except Exception as e:
        return f"Error: {str(e)}"

def start_training(model_name, epochs, batch_size, fp16_run):
    try:
        exp_dir = f"logs/{model_name}"
        os.makedirs(exp_dir, exist_ok=True)
        # Example: Simulated training command
        subprocess.run(["echo", f"Training model: {model_name} for {epochs} epochs with batch size {batch_size}. FP16: {fp16_run}"])
        return "Training started! Check logs for progress."
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
def main_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Train Voc ModeL WebUI")

        with gr.Tab("Data Preprocessing"):
            model_name = gr.Textbox(label="Model Name", placeholder="Enter model name", value="")
            dataset_folder = gr.Textbox(label="Dataset Folder Path", placeholder="/path/to/dataset")
            sample_rate = gr.Radio(label="Sample Rate", choices=["32k", "40k", "48k"], value="40k")
            is_half = gr.Checkbox(label="Memory Saving", value=True)
            new_preprocess = gr.Checkbox(label="Improved Preprocessing (Experimental)", value=False)
            percentage = gr.Slider(label="Percentage", minimum=1, maximum=100, value=3.7)
            preprocess_btn = gr.Button("Start Preprocessing")
            preprocess_output = gr.Textbox(label="Output")

            preprocess_btn.click(
                preprocess_dataset,
                inputs=[model_name, dataset_folder, sample_rate, is_half, new_preprocess, percentage],
                outputs=preprocess_output
            )

        with gr.Tab("Index Training"):
            index_version = gr.Radio(label="Version", choices=["v1", "v2"], value="v2")
            train_index_btn = gr.Button("Train Index")
            index_output = gr.Textbox(label="Output")

            train_index_btn.click(train_index, inputs=[model_name, index_version], outputs=index_output)

        with gr.Tab("Model Training"):
            epochs = gr.Textbox(label="Total Epochs", placeholder="500", value="500")
            batch_size = gr.Slider(label="Batch Size", minimum=4, maximum=32, step=4, value=8)
            fp16_run = gr.Checkbox(label="Use FP16", value=True)
            train_btn = gr.Button("Start Training")
            train_output = gr.Textbox(label="Output")

            train_btn.click(start_training, inputs=[model_name, epochs, batch_size, fp16_run], outputs=train_output)

    return demo

demo = main_interface()
demo.launch()
