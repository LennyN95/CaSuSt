# general imports
import os
import numpy as np
from tqdm import tqdm

import torchio as tio
#import SimpleITK as sitk

# local imports
from . import loading

def getSubject(input_dir, hmask_file):

    #
    assert os.path.isdir(input_dir), "input_dir must be a folder containing ct-scan dicom files for one patient"
    assert os.path.isfile(hmask_file), "hmask_file must be a nifti file containing the patients heart mask"

    # create subject
    subject = tio.Subject(
        ct_scan =  tio.ScalarImage(input_dir),  # dicom -> TODO: or better use nifti here too? does it make any difference?
        Heart = tio.LabelMap(hmask_file)        # nifti
    )

    return subject


def preprocessSubject(subject):

    assert 'Heart' in subject and 'ct_scan' in subject, "subject is malformed"
    
    # resample the subject to the spacing the model was trained on
    resample_xfm = tio.Resample((1.171875, 1.171875, 3.0))

    # apply soft-tissue windowing
    clamp_xfm = tio.Clamp(
        out_min=-135, 
        out_max=215
    )

    # map the hu (int) values to floats between -1 and 1
    rescale_xfm = tio.RescaleIntensity(
        out_min_max=(-1, 1),
        in_min_max=(-135, 215) 
    )

    # apply a per-patient z-norm on the whole ct-scan image
    znorm_xfm = tio.ZNormalization()

    # crop the patient to 149x149x53
    crop_xfm = tio.CropOrPad(
        target_shape=(149, 149, 53), 
        mask_name='Heart'
    )

    # bring the patient to canonical orientation
    tocanonical_xfm = tio.ToCanonical()

    # first transform
    t1 = tio.Compose([
        resample_xfm,
        clamp_xfm,
        rescale_xfm,
        znorm_xfm,
    ])

    # second transform
    t2 = tio.Compose([
        crop_xfm,
        tocanonical_xfm
    ])

    # apply transformations
    t1subject = t1(subject)
    t2subject = t2(t1subject)

    return t1subject, t2subject


def getData(t2subject):
    
    #
    assert 'Heart' in t2subject and 'ct_scan' in t2subject, "subject is malformed"

    # extract 2d slices
    ct_scan_np = t2subject.ct_scan.numpy() 
    heart_mask_np = t2subject.Heart.numpy() 

    # due to training conditions, mask must be 0|255 (not 0|1)
    heart_mask_np[heart_mask_np > 0] = 255

    # collect data in array corresponding to the slides
    data = []

    # iterate the slices
    for z_slice in tqdm(range(ct_scan_np.shape[3]), leave=False):
        ct_scan_np_slice = ct_scan_np[0, :, :, z_slice]
        heart_mask_np_slice = heart_mask_np[0, :, :, z_slice]
        
        # create same conditions as with training 
        # (note: predicted output must be rotated three times to revert original orientation)
        ct_scan_np_slice = np.rot90(ct_scan_np_slice)
        heart_mask_np_slice = np.rot90(heart_mask_np_slice)
        
        #
        data.append({
            'original': {
                'ct': ct_scan_np_slice,
                'heart': heart_mask_np_slice
            }
        })

    #
    return data


def addTTAug(data, t2subject, num_random_ttaugvs):
    # random, affine transformation
    afine_xfm = tio.RandomAffine(
        scales=(0.1, 0.1, 0),
        degrees=(4, 4, 0),
        translation=(5, 5, 0) # in mm
    )

    # collect the inverse-transformations for later restoring process
    ttaugs_inv = []

    # num_random_ttaugvs times, apply a random augmentation transform 
    # then iterate the data collection (each slide) and add the augmented data to the ttaug field
    for i in tqdm(range(num_random_ttaugvs)):
        
        # apply random augmentation on t2subject (cropped, canonical)
        at2subject = afine_xfm(t2subject)
        
        # get inverse transformation
        invaug_xfm = at2subject.get_inverse_transform(ignore_intensity=True)[0]
        assert isinstance(invaug_xfm, tio.transforms.augmentation.spatial.random_affine.Affine), \
            f"probaply not the inverse affine we're looking for: {type(inverse_affine)}"
        ttaugs_inv.append(invaug_xfm)
        
        # extract 2d slices
        ct_scan_np = at2subject.ct_scan.numpy() 
        heart_mask_np = at2subject.Heart.numpy() 
        
        # update data
        for z_slice in range(ct_scan_np.shape[3]):
            ct_scan_np_slice = ct_scan_np[0, :, :, z_slice]
            heart_mask_np_slice = heart_mask_np[0, :, :, z_slice]

            # create same conditions as with training 
            # (note: predicted output must be rotated three times to revert original orientation)
            ct_scan_np_slice = np.rot90(ct_scan_np_slice)
            heart_mask_np_slice = np.rot90(heart_mask_np_slice)

            #
            if not 'ttaug' in  data[z_slice]:
                data[z_slice]['ttaug'] = []
                    
            # add new ttaug for the z_slice
            data[z_slice]['ttaug'].append({
                'ct': ct_scan_np_slice,
                'heart': heart_mask_np_slice
            })

    # return
    return ttaugs_inv


def prepare(input_dir, hmask_file, output_dir, num_random_ttaugvs=20):

    # sunject
    subject = getSubject(input_dir, hmask_file)

    # preprocessing
    t1subject, t2subject = preprocessSubject(subject)

    # data preparation
    data = getData(t2subject)

    # ttaug
    ttaugs_inv = addTTAug(data, t2subject, num_random_ttaugvs)

    #
    loading.storeMilestones(output_dir, data, ttaugs_inv, t1subject, t2subject)

