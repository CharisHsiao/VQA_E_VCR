export PYTHONPATH=/home/lab/A305/yifan/vqa/vqa-exp/r2c/
export CUDNN_INCLUDE_DIR=/usr/local/cuda-10.0/include/
export CUDNN_LIB_DIR=/usr/local/cuda-10.0/lib64/
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-10.0/
nohup python train.py -params multiatt/default.json -folder saves/flagship_answer > ../out/train_q2a_log.txt 2>&1 &
tail -f ../out/train_q2a_log.txt

