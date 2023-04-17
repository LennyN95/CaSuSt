import argparse
import os
import sys
import json

# arguments
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--roi', dest='rois', type=str, nargs='+',
                    choices=['Ventricle_R', 'Ventricle_L', 'Atrium_R', 'Atrium_L', 'Coronary_Atery_R', 'Coronary_LAD', 'Coronary_Atery_CFLX'],
                    default=['Ventricle_R', 'Ventricle_L', 'Atrium_R', 'Atrium_L', 'Coronary_Atery_R', 'Coronary_LAD', 'Coronary_Atery_CFLX'],
                    help='define the regions of interest')

parser.add_argument('--output_dir', dest='output_dir', type=str,
                    help='all outputs go here')

parser.add_argument('--config', dest='config_file', type=str,
                    help='config file path')

parser.add_argument('--device', dest='device', type=str, nargs=1, choices=['cpu', 'cuda'], default='cpu',
                    help='device for training evaluation')

args = parser.parse_args()

# add parent to sys path for local sseg import
currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

print("> current directory: ", currentdir)
print("> added sys.path:    ", parentdir)

# sseg import 
from ssseg import predict

# load config 
f = open(args.config_file)
configs = json.load(f)
f.close()

# parameter
rois = args.rois
output_dir = args.output_dir
device = args.device

# manipulate configs: only keep rois that are defined 
configs = {roi: configs[roi] for roi in configs if roi in rois}

# execute
predict.predict(configs, output_dir, device='cuda')