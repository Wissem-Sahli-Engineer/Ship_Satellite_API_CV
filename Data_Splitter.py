from pathlib import Path
import os , shutil

folder_path = "data/archive/shipsnet"

ship_path = "data/ship"
no_ship_path = "data/no_ship"

if not Path(ship_path).exists():
    os.mkdir(ship_path)

if not Path(no_ship_path).exists():
    os.mkdir(no_ship_path)

for file in Path(folder_path).iterdir():
    if file.is_file():

        if file.name.startswith("1"):
            shutil.copy(file, Path(ship_path) / file.name)
        else:
            shutil.copy(file, Path(no_ship_path) / file.name)
