#!/bin/bash

export NCCL_P2P_DISABLE=1


skips=("5")
scales=("2.0")
ks=("1.0")
efilters=("9" )
base_folder="/hdd/dungda/"

for scale in "${scales[@]}"
do
for skip in "${skips[@]}"
do
for k in "${ks[@]}"
do
for efilter in "${efilters[@]}"
do
cmd="WORLD_SIZE=1 RANK=0 MASTER_IP=127.0.0.1 MASTER_PORT=29510 MARSV2_WHOLE_LIFE_STATE=0 python generate_compress_dist_zip_earlyfilter.py \
 --cond_pkl models/edm2-xxl-phema-00939524-0.015.pkl --uncond_pkl models/edm2-xs-phema-02147483-0.015.pkl \
 --zip_dir IntG/eFilter/compress_e${efilter}_s${scale}_skip${skip}_k${k} --efilter ${efilter} --k ${k} --skip ${skip} --guidance_interval '[17, 22]' --guidance_scale ${scale} --save-num 768 --fix_seed --base_folder ${base_folder}"
echo ${cmd}
eval ${cmd}
done
done
done
done

for scale in "${scales[@]}"
do
for skip in "${skips[@]}"
do
for k in "${ks[@]}"
do
for efilter in "${efilters[@]}"
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