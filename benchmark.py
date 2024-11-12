
import os
import yaml

from estimate import DepthEstimation

import helpers

import datetime


class Bag:
    def __init__(self, path):
        self.path = path

        # have roi not protected
        self.have_roi = False
        self.roi_tl = None
        self.roi_br = None

        self.metric_error = None

    def __repr__(self):
        repr = f"\nPath: {self.path}"
        if self.have_roi:
            repr += f"\nROI:    \n      {self.roi_tl}\n      {self.roi_br}"

        if self.metric_error is not None:
            repr += f"ERROR: {self.metric_error}"

        return repr
    
    def set_roi(self, pt_top_left: tuple[int], pt_bottom_right: tuple[int]):
        if not self.have_roi:
            self.roi_tl = pt_top_left
            self.roi_br = pt_bottom_right
            self.have_roi = True
        else:
            raise RuntimeError("ROI already set.")
        
    def get_roi(self):
        if self.have_roi:
            return self.roi_tl + self.roi_br
        else:
            raise ValueError(f"ROI not provided.")
        
    def set_error(self, error):
        if self.error is not None:
            raise ValueError(f"Cannot set error. Error already set.")
        
        self.error = error


class Benchmark:

    def __init__(self, datasets: list[str]):

        self.datasets = []
        
        for dataset in datasets:
            if os.path.exists(dataset):
                if os.path.isdir(dataset):
                    print(f"Reading path {dataset}")

                    data = {
                        "bags": [Bag(os.path.join(dataset, obj)) for obj in os.listdir(dataset) if obj.endswith('.bag')],
                        "camera-left": os.path.join(dataset, 'left.yaml'),
                        "camera-right": os.path.join(dataset, 'right.yaml')
                    }

                    for bag in data["bags"]:
                        if os.path.exists(f"{bag.path}-roi.yaml"):
                            with open(f"{bag.path}-roi.yaml", 'r') as stream:
                                content = yaml.full_load(stream)
                                bag.set_roi(content["roi"]["top-left"], content["roi"]["bottom-right"])

                    if not (os.path.exists(data["camera-left"]) and os.path.exists(data["camera-right"])):
                        raise FileNotFoundError("Check if camera configuration files are in {dataset} and if they are named 'left.yaml' and 'right'.yaml'") 

                    self.datasets.append(data)

                else:
                    continue
            else:
                print(f"Bad path {dataset}.")


    def evaluate(self):

        for dataset in self.datasets:

            estimator = DepthEstimation(dataset['camera-left'], dataset['camera-right'])
            
            for bag in dataset['bags']:

                estimator.set_input(bag.path)
                if bag.have_roi:
                    estimator.compute_disparity(roi=bag.get_roi())
                else:
                    estimator.compute_disparity(is_checkerboard=True, checkerboard_pattern=(10,7))
                prediction = estimator.compute_depth()

                ground_truth = float(os.path.split(bag.path)[-1].split(".")[0][:-2])

                bag.metric_error = 100*abs(prediction-ground_truth)/ground_truth

                print(f"Dataset: {bag.path}, Error: {bag.metric_error}\n\n")


        # accuracy


        # execution speed

        # memory usage

        # cpu usage
        pass


if __name__ == "__main__":

    bench = Benchmark(
        ["data/A008-01-27", 
         "data/A008-02-02"
         ]
    )

    for dataset in bench.datasets:
        print('\n', dataset['bags'])

    bench.evaluate()


    now = datetime.datetime.now()
    os.rename(helpers.path_log, '-'.join([helpers.path_log, str(now)]))