from input import extract_rgb, Meta

import numpy as np
import cv2
from scipy.signal import convolve, find_peaks, windows

from helpers import measure_execution_time, measure_memory_usage

from enum import Enum

DisparityEstimationMethod = Enum("DepthEstimationMethod", [("TM", 1), ("BM", 2)])


class DepthEstimation:

    @measure_memory_usage
    @measure_execution_time
    def __init__(self, path_to_config_left: str, path_to_config_right: str):
        """Initialize DepthEstimation object by setting left and right camera parameters."""

        self.path_to_config_left = path_to_config_left
        self.path_to_config_right = path_to_config_right

        self.meta_l = Meta(self.path_to_config_left)
        self.meta_r = Meta(self.path_to_config_right)
        self.baseline = (
            -self.meta_r.projection_matrix[0, -1] / self.meta_r.projection_matrix[0, 0]
        )

        self.path_to_bag = ""  # to be set via `set_input`
        self.raw_l, self.raw_r = None, None  # will be set in `set_input`
        self.rectified_l, self.rectified_r = None, None  # will be set in `set_input`

        self.method = None

        self.disparity = None
        self.depth = None

    @measure_memory_usage
    @measure_execution_time
    def set_input(self, path_to_bag: str):
        """Set stereo pair images and rectifies them."""

        self.raw_l, self.raw_r = None, None  # will be set in `set_input`
        self.rectified_l, self.rectified_r = None, None  # will be set in `set_input`

        self.path_to_bag = path_to_bag
        self.raw_l, self.raw_r = extract_rgb(self.path_to_bag)
        self.rectified_l, self.rectified_r = DepthEstimation.rectify(
            self.raw_l, self.meta_l, self.raw_r, self.meta_r
        )

    def set_method(self, disparity_estimation_method: DisparityEstimationMethod):
        self.method = disparity_estimation_method

    @staticmethod
    def rectify(
        left: np.array, left_meta: Meta, right: np.array, right_meta: Meta
    ) -> tuple:
        """Compute rectified stereo camera pair."""

        maps = []
        for meta, shape in zip([left_meta, right_meta], [left.shape, right.shape]):
            maps.append(
                cv2.initUndistortRectifyMap(
                    meta.camera_matrix,
                    meta.distortion_coefficients,
                    meta.rectification_matrix,
                    meta.projection_matrix,
                    (shape[1], shape[0]),
                    cv2.CV_16SC2,
                )
            )

        left_rectified = cv2.remap(left, maps[0][0], maps[0][1], cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right, maps[1][0], maps[1][1], cv2.INTER_LINEAR)

        return left_rectified, right_rectified

    @measure_memory_usage
    @measure_execution_time
    def compute_disparity(
        self,
        roi: tuple = None,
        is_checkerboard: bool = False,
        checkerboard_pattern: tuple = None,
        is_debug=False,
    ):
        """Compute disparity between rectified stereo pair. Note, have to set input before computing disparity.

        Supports two mode:
        1. Expecting an image of a checkerboard (have to set `is_checkerboard` and `checkerboard_pattern`),
        2. Expecting to provide location of an object whose distance to camera is calculated (have to set `roi`).
        """

        if self.rectified_l is None or self.rectified_r is None:
            raise RuntimeError("Have to set input images before computing disparity.")

        if self.method is None:
            raise RuntimeError("Have to set methodology to compute disparity.")

        if is_checkerboard:
            if checkerboard_pattern:

                # find checkerboard
                scale = 8
                new_shape = [dim // scale for dim in self.rectified_l.shape[:-1]]
                left_new = cv2.resize(self.rectified_l, new_shape[::-1])

                checkerboard_found, corners = cv2.findChessboardCorners(
                    left_new, (10, 7)
                )
                if not checkerboard_found:
                    raise ValueError(
                        f"Checkerboard not found. Check checkerboard pattern {checkerboard_pattern}."
                    )

                corners = corners.squeeze()
                bbox = cv2.boundingRect(corners)

                if is_debug:
                    debug_cb_img = left_new.copy()
                    for corner in corners:
                        cv2.circle(
                            debug_cb_img, corner.astype(np.int32), 2, (0, 0, 255)
                        )
                        cv2.rectangle(
                            debug_cb_img,
                            (bbox[0], bbox[1]),
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                            (0, 0, 255),
                        )

                    cv2.namedWindow("checkerboard", cv2.WINDOW_NORMAL)
                    cv2.imshow("checkerboard", debug_cb_img)
                    cv2.waitKey(0)
                    cv2.destroyWindow("checkerboard")

                # compute checkerboard offset
                sizes = np.diff(corners, axis=0)
                good_points = sizes[sizes > 1]
                cb_offset = int(good_points.mean() // 2 + 1)

                cb_left = bbox[0] - cb_offset
                cb_right = bbox[0] + bbox[2] + cb_offset
                cb_top = bbox[1] - cb_offset
                cb_bottom = bbox[1] + bbox[3] + cb_offset

                if scale > 1:
                    cb_left *= scale
                    cb_right *= scale
                    cb_top *= scale
                    cb_bottom *= scale

                cb_left = int(cb_left)
                cb_right = int(cb_right)
                cb_top = int(cb_top)
                cb_bottom = int(cb_bottom)

                roi = (cb_left, cb_top, cb_right, cb_bottom)
            else:
                raise ValueError(
                    f"Cannot use {is_checkerboard=} mode without setting {checkerboard_pattern=} properly."
                )

        elif roi is not None:
            cb_left, cb_top, cb_right, cb_bottom = roi

        else:
            raise NotImplementedError(
                "Have to support an image of checkerboard with (10,7) pattern or provide ROI of an object."
            )

        if is_debug:

            debug_rectified_l = self.rectified_l.copy()

            cb_top_left = (cb_left, cb_top)
            cb_bottom_right = (cb_right, cb_bottom)

            cv2.rectangle(
                debug_rectified_l, cb_top_left, cb_bottom_right, (0, 0, 255), 4
            )

            cv2.namedWindow("checkerboard full scale", cv2.WINDOW_NORMAL)
            cv2.imshow("checkerboard full scale", debug_rectified_l)
            cv2.waitKey(0)
            cv2.destroyWindow("checkerboard full scale")

        if self.method == DisparityEstimationMethod.BM:
            self.disparity = self.process_block_matching(roi, is_debug)
        elif self.method == DisparityEstimationMethod.TM:
            self.disparity = self.process_template_matching(roi, is_debug)
        else:
            raise ValueError(
                f"Unknown disparity estimation method provided. Got {self.method}."
            )

    @measure_memory_usage
    @measure_execution_time
    def process_template_matching(self, roi_left, is_debug):
        """Use template matching to find an object in right image"""
        rectified_gray_l = cv2.cvtColor(self.rectified_l, cv2.COLOR_RGB2GRAY)
        rectified_gray_r = cv2.cvtColor(self.rectified_r, cv2.COLOR_RGB2GRAY)

        cb_centre_x = int((roi_left[0] + roi_left[2]) * 0.5)

        template = rectified_gray_l[
            roi_left[1] : roi_left[3], roi_left[0] : roi_left[2]
        ]
        if is_debug:
            cv2.imshow("template", template)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        w, h = template.shape[::-1]

        matching_score = cv2.matchTemplate(rectified_gray_r, template, cv2.TM_SQDIFF)
        _, _, min_loc, _ = cv2.minMaxLoc(matching_score)
        top_left = min_loc

        match_centre_x = top_left[0] + w // 2

        if is_debug:

            debug_matching = self.rectified_r.copy()

            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(debug_matching, top_left, bottom_right, (255, 0, 0), 2)

            match_centre_y = top_left[1] + h // 2
            cv2.circle(
                debug_matching,
                (match_centre_x, match_centre_y),
                10,
                (255, 0, 0),
                thickness=4,
            )

            cv2.namedWindow("matching score", cv2.WINDOW_NORMAL)
            cv2.imshow("matching score", matching_score / matching_score.max())

            cv2.namedWindow("match", cv2.WINDOW_NORMAL)
            cv2.imshow("match", debug_matching)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return abs(match_centre_x - cb_centre_x)

    @measure_memory_usage
    @measure_execution_time
    def process_block_matching(self, roi_left, is_debug):

        focal_length = self.meta_l.projection_matrix[0, 0]
        baseline = (
            -self.meta_r.projection_matrix[0, -1] / self.meta_r.projection_matrix[0, 0]
        )

        left_gray_r = cv2.cvtColor(self.rectified_l, cv2.COLOR_RGB2GRAY)
        right_gray_r = cv2.cvtColor(self.rectified_r, cv2.COLOR_RGB2GRAY)

        # set min and max disparity @TODO make it a parameter
        min_depth, max_depth = 0.4, 1.75
        max_disparity = DepthEstimation.to_disparity(min_depth, focal_length, baseline)
        min_disparity = DepthEstimation.to_disparity(max_depth, focal_length, baseline)

        # Compute disparity on image pyramid
        multi_resolution = {}

        min_disparity_running = min_disparity
        max_disparity_running = max_disparity

        left_scaled_running = left_gray_r.copy()
        right_scaled_running = right_gray_r.copy()

        scale = 1
        reached = False
        if is_debug:
            print("Computing disparity on image pyramid:")
        while not reached:

            scale *= 2
            min_disparity_running *= 0.5
            max_disparity_running *= 0.5

            # coarser estimate seems to work just fine; optionally we can use some interpolation..
            left_scaled_running = left_scaled_running[::2, ::2]
            right_scaled_running = right_scaled_running[::2, ::2]
            # left_scaled_running = cv2.resize(left_scaled_running, dsize=[int(s//2) for s in left_scaled_running.shape[::-1]])
            # right_scaled_running = cv2.resize(right_scaled_running, dsize=[int(s//2) for s in right_scaled_running.shape[::-1]])

            bm_min_disparity_scaled = int(min_disparity_running // 16 * 16)
            bm_max_disparity_scaled = int((max_disparity_running // 16 + 1) * 16)

            stereo = cv2.StereoBM.create(
                numDisparities=bm_max_disparity_scaled - bm_min_disparity_scaled,
                blockSize=21,
            )
            stereo.setMinDisparity(bm_min_disparity_scaled)
            stereo.setDisp12MaxDiff(10)

            disparity = stereo.compute(left_scaled_running, right_scaled_running)

            multi_resolution[
                (scale, bm_min_disparity_scaled, bm_max_disparity_scaled)
            ] = {
                "disparity": disparity / 16,
                "left": left_scaled_running,
                "right": right_scaled_running,
            }

            if is_debug:
                print(
                    f"Computed disparity at ({scale=}, using disparity range [{bm_min_disparity_scaled}, {bm_max_disparity_scaled}]"
                )

            if bm_min_disparity_scaled == 16:
                reached = True
                continue

        # vote to select range of disparities
        total = []
        for key, value in multi_resolution.items():

            region = DepthEstimation.get_roi(
                value["disparity"], [s // key[0] for s in roi_left]
            )

            hist, bin_edges = np.histogram(
                (region * key[0]).flatten(),
                bins=int(max_disparity - min_disparity) // 4,
                range=(min_disparity, max_disparity),
            )

            # @TODO convert histogram visualization from matplotlib
            # if is_debug:
            #     plt.figure(figsize=(10,3))
            #     plt.bar((bin_edges[1:]+bin_edges[:-1])*.5, hist, width=1.)
            #     plt.xlim(bin_edges[0], bin_edges[-1])
            #     plt.show()

            if isinstance(total, list):
                total = np.zeros_like(hist)
            total += hist * key[0] * key[0]

            total = convolve(total, windows.gaussian(7, 3))
            peaks = find_peaks(total, width=6)

            bins = None
            if len(peaks[0]) == 0:
                bins = [
                    np.argmax(total),
                ]
            else:
                peak_id = np.argmax(peaks[1]["prominences"])
                bins = [
                    b
                    for b in range(
                        peaks[1]["left_bases"][peak_id],
                        peaks[1]["right_bases"][peak_id] + 1,
                    )
                ]

            disparities_count = hist[bins]
            disparities_edges = ((bin_edges[1:] + bin_edges[:-1]) * 0.5)[bins]

            disparity = (
                disparities_count * disparities_edges
            ).sum() / disparities_count.sum()

            return disparity

    @staticmethod
    def get_roi(base, roi):
        """Extract region of interest roi from a 2D matrix base. roi is
        defined in form (left, top, right, bottom)."""
        if len(base.shape) != 2:
            raise RuntimeError(
                f"Expecting two-dimensional input base. Got {base.shape}."
            )
        return base[roi[1] : roi[3], roi[0] : roi[2]]

    def compute_depth(self):
        """Return depth in mm."""
        self.depth = DepthEstimation.to_depth(
            self.disparity, self.meta_l.projection_matrix[0, 0], self.baseline
        )
        return self.depth * 1000

    @staticmethod
    def to_depth(disparity: float, focal_length: float, baseline: float):
        return baseline * focal_length / disparity

    @staticmethod
    def to_disparity(depth: float, focal_length: float, baseline: float):
        return baseline * focal_length / depth


if __name__ == "__main__":

    depth_estimation = DepthEstimation(
        path_to_config_left="data/A008-01-27/left.yaml",
        path_to_config_right="data/A008-01-27/right.yaml",
    )
    depth_estimation.set_input("data/A008-01-27/750mm.bag")
    depth_estimation.set_method(DisparityEstimationMethod.TM)

    depth_estimation.compute_disparity(
        is_checkerboard=True, checkerboard_pattern=(10, 7), is_debug=False
    )
    estimated_depth = depth_estimation.compute_depth()

    print(f"{depth_estimation.disparity=}")
    print(f"Ground truth: 750mm, prediction: {estimated_depth}")
