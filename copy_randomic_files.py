import os
import shutil
import random

from settings.settings import (
    COPY_DESTINATION_FOLDER,
    COPY_SOURCE_FOLDER,
    COPY_PERCENTAGE_TO_COPY,
    COPY_RANDOM_MODE,
    COPY_FIXED_NUMBER_TO_COPY
)


def copy_files(
        source_folder,
        destination_folder,
        percentage=None,
        fixed_number=None,
        random_mode=True
    ):

    files = [file_name for file_name in os.listdir(source_folder)
             if os.path.isfile(os.path.join(source_folder, file_name))]

    if percentage is not None:
        total_to_copy = int(len(files) * percentage / 100)
    elif fixed_number is not None:
        total_to_copy = min(fixed_number, len(files))
    else:
        raise ValueError("Either percentage or fixed_number must be provided!")

    if random_mode:
        chosen_files = random.sample(files, total_to_copy)
    else:
        chosen_files = files[:total_to_copy]

    for file_name in chosen_files:
        shutil.copy2(os.path.join(source_folder, file_name), destination_folder)

    print(f"{total_to_copy} files have been copied from {source_folder} to {destination_folder}.")


if __name__ == '__main__':
    copy_files(
        COPY_SOURCE_FOLDER,
        COPY_DESTINATION_FOLDER,
        percentage=COPY_PERCENTAGE_TO_COPY,
        fixed_number=COPY_FIXED_NUMBER_TO_COPY,
        random_mode=COPY_RANDOM_MODE
    )
