import os
import torch
from tqdm import tqdm
import numpy as np

import torchio as tio
import SimpleITK as sitk

# local imports
from . import model
from . import loading


# first inverting the ttaugmentation and then 
# taking the average over the raw output from original and all augmented versions 
def getConfidenceMaps(rois, data, ttaugs_inv, t2subject):

    roi_orig_confmaps = {}
    roi_ttaug_confmaps= {}

    for roi in rois:
        
        conf3d_collection = []

        conf3d = np.array([np.rot90(d['original'][roi]['conf'], k=3) if roi in d['original'] else np.zeros((144, 144)) for d in data]).transpose(1, 2, 0)
        roi_orig_confmaps[roi] = conf3d
        conf3d_collection.append(conf3d)

        for ttai in range(len(ttaugs_inv)):

            # rotate back (3x90deg)
            # return as numpy 3d volume: x / y / z
            conf3d_aug = np.array([np.rot90(d['ttaug'][ttai][roi]['conf'], k=3) if roi in d['ttaug'][ttai] else np.zeros((144, 144)) for d in data]).transpose(1, 2, 0)

            # add dimension 0 to match tio shape -> 0 / x / y / z
            conf3d_aug = np.expand_dims(conf3d_aug, 0)

            # create tio label map, affine is equal to the transformed subject (before any augmentations)
            confLM_aug = tio.LabelMap(tensor=conf3d_aug, affine=t2subject.Heart.affine)

            #
            invaug_xfm = ttaugs_inv[ttai]
            confLM = invaug_xfm(confLM_aug)

            #
            conf3d = confLM.numpy().squeeze(0)

            #
            conf3d_collection.append(conf3d)
            
        # mean
        conf3d_mean = np.mean(conf3d_collection, axis=0)
        roi_ttaug_confmaps[roi] = conf3d_mean

    return roi_orig_confmaps, roi_ttaug_confmaps

def getHeartMaxBoundaries2(heart_nii_npy):
   
    # dim0 (x:width)
    zi0 = np.flatnonzero([1 if heart_nii_npy[i, :, :].max() > 0 else 0 for i in range(heart_nii_npy.shape[0])])
    w = max(zi0) - min(zi0) + 1

    # dim1 (y:height)
    zi1 = np.flatnonzero([1 if heart_nii_npy[:, i, :].max() > 0 else 0 for i in range(heart_nii_npy.shape[1])])
    h = max(zi1) - min(zi1) + 1

    # dim2 (z:width)
    zi2 = np.flatnonzero([1 if heart_nii_npy[:, :, i].max() > 0 else 0 for i in range(heart_nii_npy.shape[2])])
    d = max(zi2) - min(zi2) + 1

    # center
    #center = (min(zi0) / 2 + max(zi0) / 2) , (min(zi1) / 2 + max(zi1) / 2) , (min(zi2) / 2 + max(zi2) / 2)
    #center = min(zi0) + w / 2, min(zi1) + h / 2, min(zi2) + d / 2
    center = min(zi0) + w / 2, min(zi1) + h / 2, (min(zi2) / 2 + max(zi2) / 2)

    # return
    return w, h, d, center

def getPatientCropContext(tio_heart, crop_dim, verbose=True):

    # get it as numpy array to further process. Also ditch teh first dimension which is a singleton.
    heart_nii_npy = tio_heart.numpy().squeeze(0)
    if verbose: print("[getPatientCropContext] Heart numpy shape..", heart_nii_npy.shape)

    w, h, d, center = getHeartMaxBoundaries2(heart_nii_npy)   # <-- using canonical RAS oriented heart

    if verbose: print("[getPatientCropContext.2] w, h, d..........", (w, h, d))
    if verbose: print("[getPatientCropContext.2] center...........", center)

    opadsA = np.array(center) - crop_dim / 2
    opadsB = np.array(heart_nii_npy.shape) - (center + crop_dim / 2)
    
    if verbose: print("[getPatientCropContext] opadsA.............", opadsA)
    if verbose: print("[getPatientCropContext] opadsB.............", opadsB)

    opads = tuple(zip(np.ceil(opadsA).astype(int), np.floor(opadsB).astype(int))) # ceil/floor is consistent with torchio source code
    
    if verbose: print("[getPatientCropContext] opads..............", opads)
    
    return opads

def getOriginalSourcePatient(input_dir):
    print("loading srcVol from", input_dir)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
    reader.SetFileNames(dicom_names)
    srcVol = reader.Execute()

    return srcVol

def exportVolumeAsNRRD(srcVol, conf3d_pad, filename, th=None):

    # apply th or up-scale to integer values 0-100 for confidence maps
    if th is not None:
        assert th >= 0 and th <= 1, "invalid thrashhold, must be within [0, 1]"
        output = (conf3d_pad > th).astype(np.int16)
    else:
        output = conf3d_pad * 100

    # transpose into itk data format (x/y/z -> z/y/x)
    output = output.transpose(2, 1, 0)

    # creat enew itk volume from numpy array (output)
    itkVol = sitk.GetImageFromArray(output) 
    #itkVol.CopyInformation(srcVol)  

    # copy spacing and origin from src volume (should be the dicom input)
    itkVol.SetOrigin(srcVol.GetOrigin())
    itkVol.SetSpacing((1.171875, 1.171875, 3.0))
    
    # resampling space
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkNearestNeighbor) #sitk.sitkLinear 
    resample.SetOutputDirection(srcVol.GetDirection())
    resample.SetOutputOrigin(srcVol.GetOrigin())
    resample.SetOutputSpacing(srcVol.GetSpacing())
    resample.SetSize(srcVol.GetSize())
    itkVolRS = resample.Execute(itkVol)

    # debug
    print("itkVolRS.GetSize(): ", itkVolRS.GetSize())

    # copy all information (spacing, origin, ..) from src volume (should be the dicom input)
    itkVolRS.CopyInformation(srcVol)
    
    # write file, using compression (True)
    sitk.WriteImage(itkVolRS, filename, True)

def finalize(configs, input_dir, output_dir):

    # extract rois from config keys
    rois = list(configs.keys())

    # load from preparation
    _, ttaugs_inv, t1subject, t2subject = loading.loadMilestones(output_dir)

    # load predicted data
    data = loading.loadPredictedData(output_dir)

    # process (data -> prediction 3d volumes in x/y/z)
    roi_orig_confmaps, roi_ttaug_confmaps = getConfidenceMaps(rois, data, ttaugs_inv, t2subject)

    # 
    loading.storeFinalResultsForEvaluation(output_dir, roi_orig_confmaps, roi_ttaug_confmaps)

    #
    srcVol = getOriginalSourcePatient(input_dir)

    # put in patient context
    for roi in rois:

        # get predictions (in RAS)
        org_conf3d = roi_orig_confmaps[roi].copy()
        tta_conf3d = roi_ttaug_confmaps[roi].copy()

        assert org_conf3d.shape == tta_conf3d.shape, "shape missmatch between orig and tta; separate treatment required - cannot proceed."
        
        # get opads (in LAS)
        opads = getPatientCropContext(t1subject.Heart, crop_dim=np.array(org_conf3d.shape))
        
        # from RAS prediction to LAS
        org_conf3d_LPS = np.rot90(org_conf3d, k=2) 
        tta_conf3d_LPS = np.rot90(tta_conf3d, k=2) 
        
        # apply padding
        org_conf3d_pad = np.pad(org_conf3d_LPS, pad_width=opads, mode='constant', constant_values=0) # constant_values=1 for visualization
        tta_conf3d_pad = np.pad(tta_conf3d_LPS, pad_width=opads, mode='constant', constant_values=0)

        # export raw
        np.save(os.path.join(output_dir, 'raw', f"{roi}.org.conf3d.npy"), org_conf3d_pad)
        np.save(os.path.join(output_dir, 'raw', f"{roi}.tta.conf3d.npy"), tta_conf3d_pad)

        # export 
        th = float(configs[roi]['th'])
        out_file = os.path.join(output_dir, 'nrrd', f"{roi}.org.pred.nrrd")
        exportVolumeAsNRRD(srcVol, org_conf3d_pad, out_file, th=th)

        th_tta = float(configs[roi]['th_tta'])
        out_file = os.path.join(output_dir, 'nrrd', f"{roi}.tta.pred.nrrd")
        exportVolumeAsNRRD(srcVol, tta_conf3d_pad, out_file, th=th_tta)
