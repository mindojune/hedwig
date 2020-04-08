#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=cpu_stacked
#SBATCH --mail-user=dojmin@umich.edu
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=6000m  
#SBATCH --time=00-02:30:00
#SBATCH --account=engin1
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log


# The application(s) to execute along with its input arguments and options:

echo "Running the Hierarchical Bert"
python -m models.stacked --dataset Reuters --model bert-base-uncased --max-seq-length 256 --batch-size 16 --lr 2e-5 --epochs 30

    

