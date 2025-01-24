#!/bin/bash

base_folder="/hdd/dungda/"
base_folder="./"

python generate_zip.py --cond_pkl=models/edm2-xxl-phema-00939524-0.015.pkl --uncond_pkl=models/edm2-xs-phema-02147483-0.015.pkl --zip_dir=${base_folder}/baseline --guidance_interval='[17, 22]' --guidance_scale=2.0 --verbose=True
# echo ${cmd}
# eval ${cmd}

