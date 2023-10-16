import os
import shutil

def reset_dataset():
    dataset_path = "./dataset"
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "valid")

    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(valid_path):
        shutil.rmtree(valid_path)

    os.makedirs(train_path)
    os.makedirs(valid_path)

    print("O dataset foi resetado com sucesso!")

if __name__ == "__main__":
    reset_dataset()
