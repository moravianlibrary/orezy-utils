"""Creates a ground truth dataset for the project. For each scanned book, it extracts the crop data from ScanTailor XML
files and saves it in a structured JSON format. Example output:

scan_id.json
{
  [
    "page1.tif": {
      "split": {
        "p1": (int, int),
        "p2": (int, int)
      },
      "crop": [
      {
        "rotation": float,
        "x": int,
        "y": int,
        "width": int,
        "height": int
      },
      ...
    ],
    ...
  ]
}
"""

import argparse
from xml.etree import ElementTree as ET
from typing import Dict, Any, Optional, List
import json
import os

from pyparsing import Enum


class ClassificationClass(Enum):
    page = 0
    back_cover = 1
    double_page = 2


def _lname(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _find_children(elem, name: str):
    for ch in list(elem):
        if _lname(ch.tag) == name:
            yield ch


def _find_first(elem, name: str):
    for ch in _find_children(elem, name):
        return ch
    return None


def _get_num_attr(el, candidates):
    if el is None:
        return None
    for k in candidates:
        v = el.attrib.get(k)
        if v is not None:
            try:
                return float(v)
            except ValueError:
                num = "".join(c for c in v if (c.isdigit() or c in ".-+eE"))
                try:
                    return (
                        float(num) if num not in ("", "+", "-", ".", "e", "E") else None
                    )
                except ValueError:
                    pass
    return None


def _parse_fix_orientation_angle(params_el) -> Optional[float]:
    if params_el is None:
        return None
    angle = _get_num_attr(params_el, ["rotation", "angle", "degrees"])
    if angle is not None:
        return angle
    for ch in params_el.iter():
        if ch is params_el:
            continue
        if _lname(ch.tag) in ("rotation", "angle", "degrees"):
            try:
                return float((ch.text or "").strip())
            except ValueError:
                pass
    txt = ET.tostring(params_el, encoding="unicode")
    tokens = "".join((c if c.isdigit() or c in " .-+" else " ") for c in txt).split()
    for t in tokens:
        try:
            return float(t)
        except ValueError:
            continue
    return None


def load_scantailor_crops(
    project_path: str, prefix: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    - Each source image may contain multiple logical pages (e.g., split pages).
    - Rotation = (Fix Orientation) + (Deskew), in degrees.
    - Crop comes from <content-rect> (falls back to <page-rect>).
    """
    tree = ET.parse(project_path)
    root = tree.getroot()

    # --- Resolve directories, files, images, pages ---
    dir_map = {}
    for directories in root.iter():
        if _lname(directories.tag) == "directories":
            for d in _find_children(directories, "directory"):
                dir_map[d.attrib.get("id")] = d.attrib.get("path")
            break

    file_map = {}
    for files in root.iter():
        if _lname(files.tag) == "files":
            for f in _find_children(files, "file"):
                file_map[f.attrib.get("id")] = {
                    "dirId": f.attrib.get("dirId"),
                    "name": f.attrib.get("name"),
                }
            break

    image_to_file = {}
    for images in root.iter():
        if _lname(images.tag) == "images":
            for im in _find_children(images, "image"):
                image_to_file[im.attrib.get("id")] = im.attrib.get("fileId")
            break

    page_to_image = {}
    for pages in root.iter():
        if _lname(pages.tag) == "pages":
            for p in _find_children(pages, "page"):
                page_to_image[p.attrib.get("id")] = p.attrib.get("imageId")
            break

    def page_to_filename(page_id: str) -> Optional[str]:
        image_id = page_to_image.get(page_id)
        if not image_id:
            return None
        file_id = image_to_file.get(image_id)
        if not file_id:
            return None
        f = file_map.get(file_id)
        if not f:
            return None

        filename = f["name"]
        filename = (
            filename.replace("_", "-", count=1)
            .replace(".tiff", ".jpg")
            .replace(".tif", ".jpg")
        )
        return filename

    # --- Collect per-page parameters ---
    deskew_angles = {}
    for el in root.iter():
        if _lname(el.tag) == "deskew":
            for pg in _find_children(el, "page"):
                pid = pg.attrib.get("id")
                params = _find_first(pg, "params")
                angle = _get_num_attr(params, ["angle"])
                if angle is not None:
                    deskew_angles[pid] = angle

    orientation_angles = {}
    for el in root.iter():
        if _lname(el.tag) in ("fix-orientation", "fix_orientation", "fixorientation"):
            for pg in _find_children(el, "image"):
                pid = pg.attrib.get("id")
                params = _find_first(pg, "rotation")
                angle = _get_num_attr(params, ["degrees"])
                if angle is not None:
                    orientation_angles[pid] = angle

    content_rects = {}
    for el in root.iter():
        if _lname(el.tag) == "select-content":
            for pg in _find_children(el, "page"):
                pid = pg.attrib.get("id")
                params = _find_first(pg, "params")
                if params is None:
                    continue
                crect = None
                for ch in params:
                    if _lname(ch.tag) == "content-rect":
                        crect = ch
                        break
                if crect is None:
                    for ch in params:
                        if _lname(ch.tag) == "page-rect":
                            crect = ch
                            break
                if crect is None:
                    continue
                try:
                    x = int(float(crect.attrib["x"]))
                    y = int(float(crect.attrib["y"]))
                    w = int(float(crect.attrib["width"]))
                    h = int(float(crect.attrib["height"]))
                    content_rects[pid] = (x, y, w, h)
                except (KeyError, ValueError):
                    pass

    split_pages = {}
    for el in root.iter():
        if _lname(el.tag) == "page-split":
            for img in _find_children(el, "image"):
                pid = img.attrib.get("id")
                params = _find_first(img, "params")
                if params is None:
                    continue
                pages = _find_first(params, "pages")
                if pages.get("type") != "two-pages":
                    continue
                cutter1 = _find_first(pages, "cutter1")
                if cutter1 is not None:
                    p1 = _find_first(cutter1, "p1")
                    p2 = _find_first(cutter1, "p2")
                    if p1 is not None and p2 is not None:
                        split_pages[pid] = {
                            "p1": (
                                int(float(p1.attrib["x"])),
                                int(float(p1.attrib["y"])),
                            ),
                            "p2": (
                                int(float(p2.attrib["x"])),
                                int(float(p2.attrib["y"])),
                            ),
                        }

    # --- Build grouped results: image_name -> list of page dicts ---
    grouped = {}
    for pid in page_to_image.keys():
        img_name = page_to_filename(pid)
        if not img_name:
            continue
        grouped[img_name] = {"split": None, "crop": [], "orientation": 0}

    for pid in page_to_image.keys():
        multipage_pid = str(int(pid) - 1)

        img_name = page_to_filename(pid)
        if not img_name:
            continue

        rot = 0.0
        if pid in deskew_angles:
            rot += float(deskew_angles[pid])

        if multipage_pid in orientation_angles:
            grouped[img_name]["orientation"] = int(orientation_angles[multipage_pid])

        if pid in content_rects:
            x, y, w, h = content_rects[pid]
        else:
            x = y = w = h = None

        if any([v is None for v in (x, y, w, h, rot)]):
            print(
                f"Warning: Incomplete data for page ID {pid} (image {img_name}, project {project_path})"
            )
            continue
        grouped[img_name]["crop"].append(
            {
                "rotation": rot,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            }
        )
        if multipage_pid in split_pages:
            grouped[img_name]["split"] = split_pages[multipage_pid]

    return dict(grouped)


def delete_invalid_pages(data: Dict[str, Any]) -> Dict[str, Any]:
    """Removes pages with invalid crop data from the dataset."""
    for key, value in data.items():
        # delete empty crops
        for crop in reversed(value["crop"]):
            if crop["width"] is None or crop["width"] == 0:
                print(
                    f"Removing invalid page {key} due to invalid crop width {crop['width']}."
                )
                value["crop"].remove(crop)
                if value["split"] is not None:
                    value["split"] = None
        # delete extra splits
        if value["split"] is not None and len(value["crop"]) < 2:
            print(
                f"Removing split from page {key} due to insufficient crops ({len(value['crop'])} crops)."
            )
            value["split"] = None
        # delete crop if page was meant to be a doublepage
        if (
            value["split"] is not None
            and value["crop"][0]["width"] / value["crop"][1]["width"] > 2.0
        ):
            print(
                f"Removing crop from page {key} due to likely double page with invalid split."
            )
            value["crop"] = value["crop"][:1]
            value["split"] = None
        # delete pages with no crops left
        if len(value["crop"]) == 0:
            print(f"Removing page {key} due to no valid crops left.")
            del data[key]
    return data


def is_back_cover(data: dict) -> bool:
    # Height must be similar
    if abs(data["crop"][0]["height"] - data["crop"][1]["height"]) > 100:
        return False
    # Widths must be different by at least 10%
    if data["crop"][0]["width"] / data["crop"][1]["width"] < 1.1:
        return False
    return True


def assign_classes(data: Dict[str, Any]) -> Dict[str, Any]:
    all_scans = len(list(data.values()))
    single_pages = len([d for d in list(data.values()) if d["split"] is None])
    single_ratio = single_pages / all_scans

    for key, value in data.items():
        key = key.replace(".tif", ".jpg").replace("/", "-")
        if (
            value["split"] is None
            and single_ratio < 0.9
            and value["crop"][0]["width"] > value["crop"][0]["height"]
        ):
            print(f"Double page detected in at {key}")
            value["crop"][0]["class"] = ClassificationClass.double_page.value
        elif value["split"] is not None and is_back_cover(value):
            print(f"Back cover detected in at {key}")
            value["crop"][0]["class"] = ClassificationClass.back_cover.value
            value["crop"][1]["class"] = ClassificationClass.page.value
        else:
            for i in range(len(value["crop"])):
                value["crop"][i]["class"] = ClassificationClass.page.value

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ScanTailor crop data.")
    parser.add_argument(
        "--input",
        type=str,
        default="ai-orezy-compressed",
        help="Path to the input directory containing ScanTailor projects.",
    )
    args = parser.parse_args()

    # Load all directories containing ScanTailor projects
    scan_dirs = os.listdir(args.input)
    scan_dirs = [d for d in scan_dirs if os.path.isdir(os.path.join(args.input, d))]

    for dir in sorted(scan_dirs):
        if not os.path.isdir(os.path.join(args.input, dir)):
            continue  # Skip if not a directory

        project_path = os.path.join(args.input, dir, "scanTailor")
        data = {}

        # A project contains multiple .scanTailor files called 1.scanTailor, 2, 3...
        scantailor_files = [
            f for f in os.listdir(project_path) if f.endswith(".scanTailor")
        ]
        for file in sorted(scantailor_files):
            scantailor_path = os.path.join(project_path, file)
            if "Backup" in scantailor_path:
                continue  # Skip backup files
            prefix = file.split(".")[0]
            data.update(load_scantailor_crops(scantailor_path, prefix))

        # Remove invalid pages
        data = delete_invalid_pages(data)
        # Add classes based on heuristics
        data = assign_classes(data)
        # Save the extracted data to a JSON file
        print(f"Found data for {dir}, files {scantailor_files}")
        with open(os.path.join(project_path, "metadata.json"), "w") as f:
            json.dump(data, f, indent=4)
