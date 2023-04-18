import argparse
import os
import sys

# arguments
parser = argparse.ArgumentParser(description='Process some integers.')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--input_dir', dest='input_dir', type=str,
                    help='input patient dicom directory (ct scan)')
group.add_argument('--input_file', dest='input_file', type=str,
                    help='input file in nrrd or nifti format (ct scan)')

parser.add_argument('--hmask', dest='hmask_file', type=str,
                    help='nifti file of the heart mask')

parser.add_argument('--output_dir', dest='output_dir', type=str,
                    help='all outputs go here')

parser.add_argument('--tta', dest='num_random_ttaugvs', type=int, default=0,
                    help='number of test time augmentations to be performed')

args = parser.parse_args()

# add parent to sys path for local sseg import
currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

print("> current directory: ", currentdir)
print("> added sys.path:    ", parentdir)

# sseg import 
from ssseg import prepare, loading

# parameter
input_path = args.input_dir or args.input_file
hmask_file = args.hmask_file
output_dir = args.output_dir
num_random_ttaugvs = args.num_random_ttaugvs

# execute
loading.createDirectories(output_dir)
prepare.prepare(input_path, hmask_file, output_dir, num_random_ttaugvs)