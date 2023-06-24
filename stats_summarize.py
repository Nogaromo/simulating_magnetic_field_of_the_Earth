import json
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Path to the expirement')
parser.add_argument('path', help='Path to the folder where the json file located')


if __name__ == '__main__':

    args = parser.parse_args()
    path = args.path

    full_data = []
    all_files = os.listdir(path)
    for dir in all_files:
        files_in_each_dir = os.listdir(path + "\\" + dir)
        for file in files_in_each_dir:
            if file.endswith('.json'):
                path_to_json = path + "\\" + dir + "\\" + file
                with open(path_to_json, 'r') as f:

                    data = json.load(f)

                    moon = data['moon']
                    stuck_on_earth = data['stuck_on_earth']
                    back = data['back']
                    front = data['front']

                    full_data.append([moon, stuck_on_earth, back, front])

    full_data = np.array(full_data)
    full_stats = np.sum(full_data, axis=0)

    full_stats_dict = {
        "moon": full_stats[0],
        "stuck_on_earth": full_stats[1],
        "back": full_stats[2],
        "front": full_stats[3]
                       }

    with open(path + "\\" + "full_stats.json", 'w', encoding='utf-8') as file:
        json.dump(full_stats_dict, file, ensure_ascii=False, indent=4)
