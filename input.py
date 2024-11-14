from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

import os
import yaml
import json

import numpy as np

import cv2

import argparse


def extract_rgb(filename: str) -> tuple:
    """Return stereo pair from a BAG file _filename_."""

    bag_path = Path(filename)

    assert bag_path.exists(), f"Path to BAG file {filename} not found!"
    assert filename.endswith(".bag"), f"Expecting a BAG file. Got {filename}."

    left, right = None, None

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_FOXY)

    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:

        updated_connections = []
        for connection in reader.connections:
            if "image_raw" in connection.msgtype:
                updated_connections.append(connection)

        sensor_data = {"left": None, "right": None}

        for connection, _, rawdata in reader.messages(connections=updated_connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if "image_raw" in connection.topic:
                sensor_data["left" if "left" in connection.topic else "right"] = (
                    msg.data
                )

        left = cv2.imdecode(sensor_data["left"], cv2.IMREAD_UNCHANGED)
        right = cv2.imdecode(sensor_data["right"], cv2.IMREAD_UNCHANGED)

        left = cv2.cvtColor(left, cv2.COLOR_BAYER_RGGB2RGB)
        right = cv2.cvtColor(right, cv2.COLOR_BAYER_RGGB2RGB)

    return left, right


class Meta:
    def __init__(self, path_to_yaml: str):

        self.width, self.height = None, None
        self.camera_name = None

        self.camera_matrix = None
        self.distortion_coefficients = None
        self.rectification_matrix = None
        self.projection_matrix = None

        if path_to_yaml == "":
            print("Path not provided. Metadata not set.")
        else:
            with open(path_to_yaml) as stream:
                try:
                    meta = yaml.safe_load(stream)

                    self.width = meta["image_width"]
                    self.height = meta["image_height"]
                    self.camera_name = meta["camera_name"]

                    self.camera_matrix = Meta.dict_to_np(meta["camera_matrix"])
                    self.distortion_coefficients = Meta.dict_to_np(
                        meta["distortion_coefficients"]
                    )
                    self.rectification_matrix = Meta.dict_to_np(
                        meta["rectification_matrix"]
                    )
                    self.projection_matrix = Meta.dict_to_np(meta["projection_matrix"])

                except Exception as exc:
                    print(f"Object not initialized.\n{exc}")

    @staticmethod
    def dict_to_np(dictionary: dict) -> np.array:
        return np.array(dictionary["data"]).reshape(
            (dictionary["rows"], dictionary["cols"])
        )


def metas_from_json(path_to_json: str):
    if os.path.exists(path_to_json):
        with open(path_to_json) as fp:
            d = json.load(fp)

        left_meta = Meta("")
        right_meta = Meta("")

        left_meta.camera_matrix = np.array(d["left"]["intrinsic"]).reshape((3, 3))
        left_meta.distortion_coefficients = np.array(d["left"]["distortion"]).reshape(
            (1, 5)
        )
        left_meta.rectification_matrix = np.array(d["left"]["rectification"]).reshape(
            (3, 3)
        )
        left_meta.projection_matrix = np.array(d["left"]["projection"]).reshape((3, 4))
        left_meta.width = d["width"]
        left_meta.height = d["height"]

        right_meta.camera_matrix = np.array(d["right"]["intrinsic"]).reshape((3, 3))
        right_meta.distortion_coefficients = np.array(d["right"]["distortion"]).reshape(
            (1, 5)
        )
        right_meta.rectification_matrix = np.array(d["right"]["rectification"]).reshape(
            (3, 3)
        )
        right_meta.projection_matrix = np.array(d["right"]["projection"]).reshape(
            (3, 4)
        )
        right_meta.width = d["width"]
        right_meta.height = d["height"]

        return left_meta, right_meta

    else:
        raise FileNotFoundError(f"JSON file {path_to_json} not found.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="BAG file loader.",
        description="Load stereo pair from a given BAG file.",
        epilog="Show loaded stereo pair images.",
    )
    parser.add_argument("filename")

    args = parser.parse_args()

    images = extract_rgb(args.filename)

    cv2.namedWindow("left", cv2.WINDOW_NORMAL)
    cv2.imshow("left", images[0])
    cv2.namedWindow("right", cv2.WINDOW_NORMAL)
    cv2.imshow("right", images[1])

    cv2.waitKey(0)
