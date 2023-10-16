import os
import shutil
import random

from settings.settings import COPY_SOURCE_FOLDER, DATASET_FOLDER, COPY_FIXED_NUMBER_TO_COPY, TRAIN_PERCENTAGE


def copy_files(
        source_folder,
        dataset_folder,
        fixed_number,
        random_mode=True
    ):

    files = [file_name for file_name in os.listdir(source_folder)
             if os.path.isfile(os.path.join(source_folder, file_name))]

    total_to_copy = min(fixed_number, len(files))

    if random_mode:
        chosen_files = random.sample(files, total_to_copy)
    else:
        chosen_files = files[:total_to_copy]

    # Splitting the chosen files for train and validation
    num_train = int(TRAIN_PERCENTAGE * total_to_copy)
    train_files = chosen_files[:num_train]
    valid_files = chosen_files[num_train:]

    # Define train and valid destination folders
    train_destination_folder = os.path.join(dataset_folder, 'train')
    valid_destination_folder = os.path.join(dataset_folder, 'valid')

    for file_name in train_files:
        shutil.copy2(os.path.join(source_folder, file_name), train_destination_folder)
    for file_name in valid_files:
        shutil.copy2(os.path.join(source_folder, file_name), valid_destination_folder)

    print(f"{num_train} files have been copied from {source_folder} to {train_destination_folder}.")
    print(f"{total_to_copy - num_train} files have been copied from {source_folder} to {valid_destination_folder}.")


if __name__ == '__main__':
    copy_files(
        COPY_SOURCE_FOLDER,
        DATASET_FOLDER,
        fixed_number=COPY_FIXED_NUMBER_TO_COPY,
        random_mode=True
    )
