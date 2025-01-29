#!/bin/bash

#!/bin/bash

#PBS -q gpuvolta
#PBS -P jp09
#PBS -l walltime=48:00:00
#PBS -l mem=256GB
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l jobfs=128GB
#PBS -l wd
#PBS -l storage=scratch/jp09+scratch/li96
#PBS -M adin6536@uni.sydney.edu.au
#PBS -o output_nci/compress_guidance_edm2_skip6_scale0.4_k3.txt
#PBS -e errors/edm2_skip6_scale1.4_k3.txt

module load use.own
module load python3/3.9.2
module load gint


skips=("6")
scales=("0.4")
ks=("3.0")
base_folder="/scratch/li96/dd9648/"


for scale in "${scales[@]}"
do
for skip in "${skips[@]}"
do
for k in "${ks[@]}"
do
cmd="WORLD_SIZE=1 RANK=0 MASTER_IP=127.0.0.1 MASTER_PORT=29510 MARSV2_WHOLE_LIFE_STATE=0 python generate_compress_dist_zip.py \
 --cond_pkl models/edm2-xxl-phema-00939524-0.015.pkl --uncond_pkl models/edm2-xs-phema-02147483-0.015.pkl \
 --zip_dir IntG/compress_s${scale}_skip${skip}_k${k} --k ${k} --skip ${skip} --guidance_interval '[17, 22]' --guidance_scale ${scale} --save-num 768 --fix_seed --base_folder ${base_folder}"
echo ${cmd}
eval ${cmd}
done
done
done

for scale in "${scales[@]}"
do
for skip in "${skips[@]}"
do
for k in "${ks[@]}"
do
cmd="python calculate_metrics_args.py --gen_zip ${base_folder}/IntG/compress_s${scale}_skip${skip}_k${k}/reference/samples_50000x512x512x3.npz --ref_path ${base_folder}/reference/img512.pkl --metric fid --base_folder ${base_folder}"
echo ${cmd}
eval ${cmd}

cmd="python calculate_metrics_args.py --gen_zip ${base_folder}/IntG/compress_s${scale}_skip${skip}_k${k}/reference/samples_50000x512x512x3.npz --ref_path ${base_folder}/reference/img512.pkl --metric fd_dinov2"
echo ${cmd}
# eval ${cmd}
done
done
done


# for scale in "${scales[@]}"
# do
# for skip in "${skips[@]}"
# do
# cmd="WORLD_SIZE=1 RANK=0 MASTER_IP=127.0.0.1 MASTER_PORT=29510 MARSV2_WHOLE_LIFE_STATE=0 python generate_compress_dist.py \
#  --cond_pkl models/edm2-xxl-phema-00939524-0.015.pkl --uncond_pkl models/edm2-xs-phema-02147483-0.015.pkl \
#  --zip_dir IntG/compress_${skip} --skip ${skip} --guidance_interval '[17, 22]' --guidance_scale ${scale} --save-num 2000 --fix_seed --base_folder ${base_folder}"
# echo ${cmd}
# eval ${cmd}
# done
# done