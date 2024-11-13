
from input import extract_rgb, Meta

import numpy as np
import cv2

from helpers import measure_execution_time, measure_memory_usage


class DepthEstimation:

    @measure_memory_usage
    @measure_execution_time
    def __init__(self, path_to_config_left: str, path_to_config_right: str):


        self.path_to_config_left = path_to_config_left
        self.path_to_config_right = path_to_config_right

        self.meta_l = Meta(self.path_to_config_left)
        self.meta_r = Meta(self.path_to_config_right)
        self.baseline = -self.meta_r.projection_matrix[0,-1] / self.meta_r.projection_matrix[0,0]


        self.path_to_bag = "" # to be set via `set_input`
        self.raw_l, self.raw_r = None, None # will be set in `set_input`
        self.rectified_l, self.rectified_r = None, None # will be set in `set_input`

        self.disparity = None
        self.depth = None

    @measure_memory_usage
    @measure_execution_time
    def set_input(self, path_to_bag: str):

        self.path_to_bag = path_to_bag
        self.raw_l, self.raw_r = extract_rgb(self.path_to_bag)
        self.rectified_l, self.rectified_r = DepthEstimation.rectify(self.raw_l, self.meta_l,
                                                                     self.raw_r, self.meta_r)


    @staticmethod
    def rectify(left: np.array, left_meta: Meta, right: np.array, right_meta: Meta) -> tuple:
        """Compute rectified stereo camera pair."""

        maps = []
        for meta, shape in zip([left_meta, right_meta], [left.shape, right.shape]):
            maps.append(cv2.initUndistortRectifyMap(meta.camera_matrix, 
                                                    meta.distortion_coefficients, 
                                                    meta.rectification_matrix,
                                                    meta.projection_matrix,
                                                    (shape[1], shape[0]), 
                                                    cv2.CV_16SC2))

        left_rectified = cv2.remap(left, maps[0][0], maps[0][1], cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right, maps[1][0], maps[1][1], cv2.INTER_LINEAR)
        
        return left_rectified, right_rectified
    

    @measure_memory_usage
    @measure_execution_time
    def compute_disparity(self, roi: tuple=None, is_checkerboard: bool=False, checkerboard_pattern: tuple=None, is_debug=False):
        if is_checkerboard:
            if checkerboard_pattern:

                # find checkerboard
                scale = 8
                new_shape = [dim // scale for dim in self.rectified_l.shape[:-1]]
                left_new = cv2.resize(self.rectified_l, new_shape[::-1])

                checkerboard_found, corners = cv2.findChessboardCorners(left_new, (10,7))
                if not checkerboard_found:
                    raise ValueError(f"Checkerboard not found. Check checkerboard pattern {checkerboard_pattern}.")

                corners = corners.squeeze()    
                bbox = cv2.boundingRect(corners)

                if is_debug:
                    debug_cb_img = left_new.copy()
                    for corner in corners:
                        cv2.circle(debug_cb_img, corner.astype(np.int32), 2, (0,0,255))
                        cv2.rectangle(debug_cb_img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,0,255))

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
            else:
                raise ValueError(f"Cannot use {is_checkerboard=} mode without setting {checkerboard_pattern=} properly.")
        
        elif roi is not None:
            cb_left, cb_top, cb_right, cb_bottom = roi   

        else:
            raise NotImplementedError("Have to support an image of checkerboard with (10,7) pattern or provide ROI of an object.")



        cb_centre_x = int((cb_left + cb_right)*.5)

        if is_debug:

            debug_rectified_l = self.rectified_l.copy()

            cb_top_left = (cb_left, cb_top)
            cb_bottom_right = (cb_right, cb_bottom)

            cv2.rectangle(debug_rectified_l, cb_top_left, cb_bottom_right, (0,0,255), 4)

            cv2.namedWindow("checkerboard full scale", cv2.WINDOW_NORMAL)
            cv2.imshow("checkerboard full scale", debug_rectified_l)
            cv2.waitKey(0)
            cv2.destroyWindow("checkerboard full scale")

        # use template matching to find checkerboard in right image
        rectified_gray_l = cv2.cvtColor(self.rectified_l, cv2.COLOR_RGB2GRAY)
        rectified_gray_r = cv2.cvtColor(self.rectified_r, cv2.COLOR_RGB2GRAY)

        template = rectified_gray_l[cb_top:cb_bottom, cb_left:cb_right]
        if is_debug:
            cv2.imshow("template", template)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        w, h = template.shape[::-1]

        matching_score = cv2.matchTemplate(rectified_gray_r, template, cv2.TM_SQDIFF)
        _, _, min_loc, _ = cv2.minMaxLoc(matching_score)
        top_left = min_loc

        match_centre_x = top_left[0] + w//2

        if is_debug:

            debug_matching = self.rectified_r.copy()

            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(debug_matching,top_left, bottom_right, (255,0,0), 2)

            match_centre_y = top_left[1] + h//2                    
            cv2.circle(debug_matching, (match_centre_x,match_centre_y), 10, (255,0,0), thickness=4)

            cv2.namedWindow("matching score", cv2.WINDOW_NORMAL)
            cv2.imshow("matching score", matching_score/matching_score.max())

            cv2.namedWindow("match", cv2.WINDOW_NORMAL)
            cv2.imshow("match", debug_matching)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.disparity = abs(match_centre_x-cb_centre_x)

    
    def compute_depth(self):
        """Return depth in mm."""
        self.depth = DepthEstimation.to_depth(self.disparity, self.meta_l.projection_matrix[0,0], self.baseline)
        return self.depth*1000

    @staticmethod
    def to_depth(disparity: float, focal_length: float, baseline: float):
        return baseline*focal_length / disparity
    

    @staticmethod
    def to_disparity(depth: float, focal_length: float, baseline: float):
        return baseline*focal_length / depth
    

if __name__ == "__main__":

    depth_estimation = DepthEstimation(
        path_to_config_left='data/left.yaml',
        path_to_config_right='data/right.yaml'
    )

    depth_estimation.compute_disparity(is_checkerboard=True, checkerboard_pattern=(10,7), is_debug=False)
    estimated_depth = depth_estimation.compute_depth()

    print(f"{depth_estimation.disparity=}")
    print(f"Ground truth: 750mm, prediction: {estimated_depth}")