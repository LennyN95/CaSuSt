import os
import torch
from tqdm import tqdm
import numpy as np

# local imports
from . import model
from . import loading


def getModel(weight_path, device):
    assert os.path.isfile(weight_path), "weight file not found"

    # load state dict
    checkpoint = torch.load(weight_path, map_location=torch.device(device)) 

    # instantiate mdoel and load weights
    unet = model.UNET(in_channels=1, out_channels=1).to(device)
    unet.load_state_dict(checkpoint)

    # eval mode
    _ = unet.eval()

    # return model instance
    return unet

def makeDividableBy16(arr, verbose=True):
    c = arr.shape[0] % 16 / 2
    ci, cj = int(np.floor(c)), -int(np.ceil(c))
    if verbose: print(f'change shape from {arr.shape} to {arr[ci:cj,ci:cj].shape} (correction: {ci}:{cj})')
    return arr[ci:cj,ci:cj]

def predictROI(data, roi, config, device):
    
    print("processing: ", roi)
    
    # config
    weight_path = config['weight_path']
    masking = config['masking']
    require_in_heart = config['masking']
    
    print("config:")
    print(" - masking: ", "on" if masking else "off")
    
    # instantiate mdoel and load weights
    unet = getModel(weight_path, device)
    
    # iterate all images
    for i, d in enumerate(tqdm(data, leave=False)):
    
        #
        versions = [d['original']]
        
        if 'ttaug' in d:
            versions += d['ttaug']
    
        for vd in versions:
    
            # get single input ct-slice
            image = vd['ct'].copy()

            # masking
            if masking:
                image = image * vd['heart']

            # check if slice is within the heart (masking in z-dimension)
            is_in_heart = vd['heart'].max() > 0
            if require_in_heart and not is_in_heart:
                continue

            # div by 16
            image = makeDividableBy16(image, verbose=(i==0))

            # numpy -> tensor
            image = torch.tensor(image)

            # add batch and channel dimension
            image = image.unsqueeze(0).unsqueeze(0).to(device)

            # comput eprediction
            with torch.no_grad():
                prediction = torch.sigmoid(unet(image))

            # prediction mask
            pred_mask = (prediction > 0.5)[0, 0, :, :].float().cpu().numpy()
            conf_mask = prediction[0, 0, :, :].float().cpu().numpy()

            # update d
            vd[roi] = {
                'image': image[0, 0].cpu(),
                'conf': conf_mask,
                'pred': pred_mask
            }

def predict(configs, output_dir, device):

    # sanity check
    for roi in configs:
        assert 'weight_path' in configs[roi] and os.path.isfile(configs[roi]['weight_path']), f"weight path missing in config for {roi}"
        assert 'masking' in configs[roi], f"masking not determined for {roi}"

    # all rois to encounter
    rois = list(configs.keys())

    # load data
    data = loading.loadPreparedData(output_dir)

    #
    for roi in tqdm(rois):
        predictROI(data, roi, config=configs[roi], device=device)

    #
    loading.storePredictions(output_dir, data)