from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

import cv2

import argparse


def extract_rgb(filename: str) -> tuple:
    """Return stereo pair from a BAG file _filename_.
    """

    bag_path = Path(filename)

    assert bag_path.exists(), f"Path to BAG file {filename} not found!"
    assert filename.endswith('.bag'), f"Expecting a BAG file. Got {filename}."
     
    left, right = None, None

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_FOXY)

    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:

        updated_connections = []
        for connection in reader.connections:
            if 'image_raw' in connection.msgtype:
                updated_connections.append(connection)

        sensor_data = {
            'left': None,
            'right': None
            }
        
        for connection, _, rawdata in reader.messages(connections=updated_connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if 'image_raw' in connection.topic:
                sensor_data['left' if 'left' in connection.topic else 'right'] = msg.data

        left = cv2.imdecode(sensor_data['left'], cv2.IMREAD_UNCHANGED)
        right = cv2.imdecode(sensor_data['right'], cv2.IMREAD_UNCHANGED)

        left = cv2.cvtColor(left, cv2.COLOR_BAYER_RGGB2RGB)
        right = cv2.cvtColor(right, cv2.COLOR_BAYER_RGGB2RGB)
        
     
    return left, right


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog="BAG file loader.",
        description="Load stereo pair from a given BAG file.",
        epilog="Show loaded stereo pair images."
    )
    parser.add_argument('filename')

    args = parser.parse_args()

    left, right = extract_rgb(args.filename)

    cv2.namedWindow("left", cv2.WINDOW_NORMAL)
    cv2.imshow("left", left)
    cv2.namedWindow("right", cv2.WINDOW_NORMAL)
    cv2.imshow("right", right)

    cv2.waitKey(0)
