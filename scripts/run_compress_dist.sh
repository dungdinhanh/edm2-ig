#!/bin/bash

export NCCL_P2P_DISABLE=1

base_folder="/hdd/dungda/"
base_folder="./"

cmd="WORLD_SIZE=1 RANK=0 MASTER_IP=127.0.0.1 MASTER_PORT=29510 MARSV2_WHOLE_LIFE_STATE=0 python generate_compress_dist.py \
 --cond_pkl models/edm2-xxl-phema-00939524-0.015.pkl --uncond_pkl models/edm2-xs-phema-02147483-0.015.pkl \
 --zip_dir IntG/compress_test --guidance_interval '[17, 22]' --guidance_scale 2.0  --fix_seed --base_folder ${base_folder}"
echo ${cmd}
eval ${cmd}

