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

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--input_dir', dest='input_dir', type=str,
                    help='input patient dicom directory (ct scan)')
group.add_argument('--input_file', dest='input_file', type=str,
                    help='input file in nrrd or nifti format (ct scan)')

parser.add_argument('--output_dir', dest='output_dir', type=str,
                    help='all outputs go here')

parser.add_argument('--config', dest='config_file', type=str,
                    help='config file path')

args = parser.parse_args()

# add parent to sys path for local sseg import
currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

print("> current directory: ", currentdir)
print("> added sys.path:    ", parentdir)

# load config 
f = open(args.config_file)
configs = json.load(f)
f.close()

# sseg import 
from ssseg import finalize

# parameter
rois = args.rois
input_path = args.input_dir or args.input_file
output_dir = args.output_dir

# execute
finalize.finalize(configs, input_path, output_dir)