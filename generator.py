import argparse
import csv
import datetime
import os

import pandas as pd
from datasets import Dataset
from datasets import load_dataset

SERIALIZE_IN_CHUNKS = 10000
FEATHER_FORMAT = "ftr"
# Block the following formats.
IMAGE = ["png", "jpg", "jpeg", "gif"]
VIDEO = ["mp4", "jfif"]
DOC = [
    "key",
    "PDF",
    "pdf",
    "docx",
    "xlsx",
    "pptx",
]
AUDIO = ["flac", "ogg", "mid", "webm", "wav", "mp3"]
ARCHIVE = ["jar", "aar", "gz", "zip", "bz2"]
MODEL = ["onnx", "pickle", "model", "neuron"]
OTHERS = [
    "npy",
    "index",
    "inv",
    "index",
    "DS_Store",
    "rdb",
    "pack",
    "idx",
    "glb",
    "gltf",
    "len",
    "otf",
    "unitypackage",
    "ttf",
    "xz",
    "pcm",
    "opus",
]
ANTI_FOMATS = tuple(IMAGE + VIDEO + DOC + AUDIO + ARCHIVE + OTHERS)
# Allow the following formats.
ALLOWED_FILE_FORMATS = ("py", "jsx", "js", "java", "php", "dart", "md")


def extract_code(directory_path: str, output_csv_path: str):
    """
    Read the folders, sub-folders and files to extract code. Storing them in a csv format.
    Currently supporting the below formats -
    .py
    .jsx
    .js
    .java
    .php
    .dart
    .md
    """

    # Create a list to store the data
    data = []

    is_allowed = False

    # Supported file formats
    lfile_formats = ('.py', '.jsx', 'js', '.java', '.php', '.dart', '.md')

    # Iterate through all files in the directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(lfile_formats):
                file_path = os.path.join(root, filename)

                # Read the file content
                with open(file_path, 'r') as f:
                    content = f.read()

                # Add the data to the list
                data.append([len(data), file_path, content])

    # Write the data to a CSV file
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    os.mkdir(f"./datasets/{timestamp}")
    dataset_name = f"./datasets/{timestamp}/dataset_{timestamp}.csv"
    with open(dataset_name, 'w', newline='') as csvfile:
        fieldnames = ['file_path', 'content', 'index']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, row in enumerate(data):
            writer.writerow({'index': i, 'file_path': row[1], 'content': row[2]})

    dataframe = pd.read_csv(dataset_name)
    dataset = Dataset.from_pandas(dataframe)
    dataset = dataset.train_test_split(
        test_size=0.1, seed=42, shuffle=True
    )

    os.mkdir(f"./datasets/{timestamp}/data")
    dataset.save_to_disk(f"./datasets/{timestamp}/data")

    print(f"Data written to {output_csv_path}")


def csv_to_dataframe(csv_filepath: str):
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    dataset_name = f"dataset_{timestamp}"
    dataframe = pd.read_csv(csv_filepath)
    dataset = Dataset.from_pandas(dataframe)
    dataset = dataset.train_test_split(
        test_size=0.1, seed=42, shuffle=True
    )

    dataset.save_to_disk(f"./datasets/{dataset_name}")


def load_data(dataset_folder: str):
    dataset = load_dataset("arrow", data_files={
        'train': dataset_folder+'/data/train/train.arrow',
        'test': dataset_folder+'/data/test/test.arrow'
    })
    print(f"Dataset loaded from {dataset_folder}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='PROG',
        description='''This tool can be used to extract code from 
        git repositories to parse it into a dataframe. These Datasets could be saved as csv and as .arrow Dataset
         The Dataframe can be used to finetune OpenSource LLMs''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-d', type=str, default="/home",
                        required=True, help='git repository root path')
    parser.add_argument('-o', default="/tmp",
                        required=True, help='csv output path')
    parser.add_argument('-m', choices=["csv", "train", "both"], required=True, default="csv")
    parser.add_argument('-mo', help='model you want to finetune only needed ')

    args = parser.parse_args()

    os.makedirs(f"./datasets", exist_ok=True)

    if args.m == "csv":
        extract_code(args.d, args.o)
    elif args.m == "train":
        load_dataset(args.o)
    elif args.m == "both":
        extract_code(args.d, args.o)
        csv_to_dataframe(args.output_csv_path)
