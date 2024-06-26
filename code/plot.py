import os
import numpy as np
import itertools

import matplotlib.pyplot as plt

def load_metrics(directory):
    """
    Load metrics from given directory. Assumes specific naming conventions for files.
    """
    metrics = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if "epoch_train_losses" in filename:
            metrics['train_loss'] = np.loadtxt(filepath)
        elif "epoch_val_losses" in filename:
            metrics['val_loss'] = np.loadtxt(filepath)
        elif "avg_class_acc" in filename:
            metrics['avg_class_acc'] = np.loadtxt(filepath)
        elif "total_acc" in filename:
            metrics['total_acc'] = np.loadtxt(filepath)
    return metrics

def plot_metrics(metrics_dict):
    """
    Plot metrics from all models, using different line styles (solid, dashed, etc.) for each model within each metric plot.
    All plots will be in black and white.
    """
    labels = {'train_loss': 'Training Loss', 'val_loss': 'Validation Loss', 
              'avg_class_acc': 'Average Class Accuracy', 'total_acc': 'Total Accuracy'}
    
    # Line styles to differentiate models. Add more styles if you have more than four models.
    line_styles = ['-', '--', '-.', ':']
    
    # Determine number of epochs based on the length of the first encountered metric for proper X-axis scaling
    num_epochs = next(iter(next(iter(metrics_dict.values())).values())).shape[0]
    epochs = np.arange(1, num_epochs + 1)

    for metric_name in labels.keys():
        plt.figure()  # Create a new plot for each metric
        for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
            if metric_name in metrics:
                # Cycle through line styles based on model index
                line_style = line_styles[idx % len(line_styles)]
                plt.plot(epochs, metrics[metric_name], line_style, label=f"{labels[metric_name]} ({model_name})", alpha=0.7)
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.title(f"{labels[metric_name]}")
        plt.legend()
        plt.tight_layout()  # Adjust layout to not cut off labels
        # Don't show just save
        plt.savefig(f'{metric_name}.png')
        # plt.show()

def main(root_directory):
    """
    Traverse the given root directory to load metrics from each model's directory and plot them.
    """
    all_metrics = {}
    for model_dir in os.listdir(root_directory):
        dir_path = os.path.join(root_directory, model_dir)
        if os.path.isdir(dir_path):  # Ensure it's a directory
            model_metrics = load_metrics(dir_path)
            if model_metrics:
                all_metrics[model_dir] = model_metrics
    
    plot_metrics(all_metrics)

if __name__ == "__main__":
    root_directory = "./content/models"  # Adjust as necessary to your models' root directory
    main(root_directory)
