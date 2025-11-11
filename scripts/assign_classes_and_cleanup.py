from collections import defaultdict
from enum import Enum
import os
import json


class ClassificationClass(Enum):
    page = 0
    back_cover = 1
    double_page = 2

def is_valid_page(data: dict) -> bool:
    if data["crop"][1]["width"] == 0:
        return False
    if data["crop"][0]["width"] / data["crop"][1]["width"] > 2.0:
        return False
    return True

def is_back_cover(data: dict) -> bool:
    # Height must be similar
    if abs(data["crop"][0]["height"] - data["crop"][1]["height"]) > 100:
        return False
    # Widths must be different by at least 10%
    if data["crop"][0]["width"] / data["crop"][1]["width"] < 1.1:
        return False
    return True

def walk_through_dataset():

    results = {"invalid": defaultdict(list), "back_covers": defaultdict(list), "double_pages": defaultdict(list)}
    files = os.listdir("datasets/scantailor_data/")
    
    for file in files:
        if not file.endswith(".json"):
            continue
        filepath = os.path.join("datasets/scantailor_data/", file)
        with open(filepath, "r") as f:
            data = f.read()
            data = json.loads(data)
        
        all_scans = len(list(data.values()))
        single_pages = len(
            [d for d in list(data.values()) if d["split"] is None]
        )
        single_ratio = single_pages / all_scans
        #print(f"{file}: {single_ratio * 100}% single pages")

        for key, value in data.items():
            if value["split"] is None:
                if single_ratio < 0.9 and value["crop"][0]["width"] > value["crop"][0]["height"]:
                    results["double_pages"][file] += [key]
                    print(f"Double page detected in {file} at {key}")
                    continue
            else:   
                if not is_valid_page(value):
                    results["invalid"][file] += [key]
                    print(f"Invalid page detected in {file} at {key}")
                    continue
                
                if is_back_cover(value):
                    results["back_covers"][file] += [key]
                    print(f"Back cover detected in {file} at {key}")
                    continue

    return results

walk_through_dataset()
