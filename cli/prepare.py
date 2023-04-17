import argparse
import os
import sys

# arguments
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--input_dir', dest='input_dir', type=str,
                    help='input patient dicom directory')

parser.add_argument('--hmask', dest='hmask_file', type=str,
                    help='nifti file of th eheart mask')

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
input_dir = args.input_dir
hmask_file = args.hmask_file
output_dir = args.output_dir
num_random_ttaugvs = args.num_random_ttaugvs

# execute
loading.createDirectories(output_dir)
prepare.prepare(input_dir, hmask_file, output_dir, num_random_ttaugvs)