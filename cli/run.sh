# zsh cli/run.sh ../data/patients/pz8/ run_chain_out/ ../data/nifti/pz8/mask_cuore-in-toto.nii.gz 20

echo "SSSEG Export Helper"
echo "expected parameters: input_dir | output_dir | hmask | tta | config"
echo " - input_dir: input dicom folder of original pCT scans."
echo " - output_dir: output directory where temporary and final results are stored."
echo " - hmask: file path to the heart mask in nifti format."
echo " - tta: number of random transformations for test time augmentation."
echo " - config: file path to the configuration file in json format."

echo "This script expects pytorch_env for pre- and postprocessing and debug_env for evaluation"
echo "The following environments are available:"
conda env list

# create output directory
mkdir $PWD/$2

echo "1 | Preprocessing"

/opt/conda/envs/pytorch_env/bin/python ${0:a:h}/prepare.py --input_dir=$1 --output_dir=$2 --hmask=$3 --tta=$4

echo "2 | Prediction"

/opt/conda/envs/debug_env/bin/python ${0:a:h}/predict.py --output_dir=$2 --device=cuda  --config=$5

echo "3 | Finalization / Export"

/opt/conda/envs/pytorch_env/bin/python ${0:a:h}/finalize.py --input_dir=$1 --output_dir=$2  --config=$5

