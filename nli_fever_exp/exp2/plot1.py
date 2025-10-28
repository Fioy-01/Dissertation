import os
import pandas as pd
import matplotlib.pyplot as plt

csv_folder = "."

csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]
dataframes = {}

for file in csv_files:
    model_name = file.replace(".csv", "")
    df = pd.read_csv(os.path.join(csv_folder, file))
    dataframes[model_name] = df

metrics = ["accuracy", "macro_f1", "precision", "recall"]

for metric in metrics:
    plt.figure(figsize=(10, 6))
    for model_name, df in dataframes.items():
        if metric in df.columns:
            plt.plot(df["labeled_size"], df[metric], marker="o", label=model_name)
    plt.title(f"{metric.capitalize()} vs. Labeled Size")
    plt.xlabel("Labeled Size")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{metric}_plot.png")
    plt.close()
