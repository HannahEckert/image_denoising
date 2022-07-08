from __future__ import print_function, unicode_literals, absolute_import, division


from csbdeep.models import CARE
from tqdm import tqdm
from segmentation.util.utils_pipeline import _create_or_continue_zarr
from wbfm.utils.general.preprocessing.bounding_boxes import get_bounding_box_via_gaussian_blurring
from segmentation.util.utils_pipeline import get_volume_using_bbox
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.finished_project_data import ProjectData


def csbdeep_predict_volume(volume, model):
    # volume: takes np.array of dim (z,y,x)
    # model_name:str folder name of predicted model
    # path: to dir that contains the model

    box = get_bounding_box_via_gaussian_blurring(volume)
    boxed_volume = volume[:, box[0]:box[2], box[1]:box[3]]

    axes = 'ZYX'
    restored = model.predict(boxed_volume, axes)
    restored_big = volume.copy()
    restored_big[:, box[0]:box[2], box[1]:box[3]] = restored

    return restored_big


def csbdeep_predict(video, fname, model_name='my_model',
                    path="/scratch/neurobiology/zimmer/hannah/repos/CSBDeep/examples/denoising3D/models"):
    restored_video = _create_or_continue_zarr(fname + ".zarr", num_frames=video.shape[0], num_slices=video.shape[1],
                                              x_sz=video.shape[2], y_sz=video.shape[3], mode='w-')

    model = CARE(config=None, name=model_name, basedir=path)
    for i in tqdm(range(video.shape[0])):
        volume = csbdeep_predict_volume(video[i, :, :], model=model)
        restored_video[i, :, :, :] = volume

    return restored_video


def csbdeep_predict_using_config(project_cfg, fname_for_saving, model_name='my_model',
                                 path="/scratch/neurobiology/zimmer/hannah/repos/CSBDeep/examples/denoising3D/models"):
    # Open the file
    project_dat = ProjectData.load_final_project_data_from_config(project_cfg)
    video_dat = project_dat.red_data

    csbdeep_predict(video_dat, fname=fname_for_saving, model_name='my_model',
                    path="/scratch/neurobiology/zimmer/hannah/repos/CSBDeep/examples/denoising3D/models")
