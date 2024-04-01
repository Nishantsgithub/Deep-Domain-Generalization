#!/bin/bash
#$ -N DIFEX_GPU
#$ -l gpu=1
#$ -P rse
#$ -q rse.q
#$ -l rmem=18G # size of memory requested
#$ -o ../output/run_with_gpu.txt  # This is where your output and errors are logged
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M youremail@shef.ac.uk # notify you by email, remove this line if you don't want to be notified
#$ -m ea # email you when it finished or aborted
#$ -cwd # run job from current directory

module load libs/CUDA
module load apps/python/conda

source activate dgml

python train.py --data_dir "C:\Users\kulde\OneDrive\Desktop\COM6911\Data\PACS\kfold" --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output ../output/difexpacs/2-0/ --test_envs 0 --dataset PACS --algorithm DIFEX --alpha 0.01 --beta 0.001 --lam 0.1 --disttype 2-norm
python train.py --data_dir "C:\Users\kulde\OneDrive\Desktop\COM6911\Data\PACS\kfold" --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output ../output/difexpacs/2-1/ --test_envs 1 --dataset PACS --algorithm DIFEX --alpha 0.001 --beta 0.1 --lam 0.01 --disttype 2-norm
python train.py --data_dir "C:\Users\kulde\OneDrive\Desktop\COM6911\Data\PACS\kfold" --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output ../output/difexpacs/2-2/ --test_envs 2 --dataset PACS --algorithm DIFEX --alpha 0.1 --beta 0.01 --lam 0.001 --disttype 2-norm
python train.py --data_dir "C:\Users\kulde\OneDrive\Desktop\COM6911\Data\PACS\kfold" --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output ../output/difexpacs/2-3/ --test_envs 3 --dataset PACS --algorithm DIFEX --alpha 0.001 --beta 0.01 --lam 0.1 --disttype 2-norm

python train.py --data_dir "C:\Users\kulde\OneDrive\Desktop\COM6911\Data\PACS\kfold" --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output ../output/difexpacs/n1n-0/ --test_envs 0 --dataset PACS --algorithm DIFEX --alpha 0.01 --beta 0.1 --lam 0.001 --disttype norm-1-norm
python train.py --data_dir "C:\Users\kulde\OneDrive\Desktop\COM6911\Data\PACS\kfold" --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output ../output/difexpacs/n1n-1/ --test_envs 1 --dataset PACS --algorithm DIFEX --alpha 0.1 --beta 10 --lam 0.1 --disttype norm-1-norm
python train.py --data_dir "C:\Users\kulde\OneDrive\Desktop\COM6911\Data\PACS\kfold" --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output ../output/difexpacs/n1n-2/ --test_envs 2 --dataset PACS --algorithm DIFEX --alpha 0.01 --beta 0.5 --lam 0.01 --disttype norm-1-norm
python train.py --data_dir "C:\Users\kulde\OneDrive\Desktop\COM6911\Data\PACS\kfold" --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output ../output/difexpacs/n1n-3/ --test_envs 3 --dataset PACS --algorithm DIFEX --alpha 0.01 --beta 1 --lam 1 --disttype norm-1-norm
