import argparse
import os

from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from functions_for_csbdeep_image_denoising import csbdeep_predict_using_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean using a neural network')
    parser.add_argument('--project_path', default=None,
                        help='path to config file')
    args = parser.parse_args()
    project_path = args.project_path

    cfg = ModularProjectConfig(project_path)
    saving_dir = cfg.get_visualization_dir()
    fname_for_saving = os.path.join(saving_dir, 'restored_dat')

    csbdeep_predict_using_config(project_path, fname_for_saving)

