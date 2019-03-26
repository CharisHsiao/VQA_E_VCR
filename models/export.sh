# source actiavte nsvqa
export PYTHONPATH=/home/lab/A305/yifan/vqa/vqa-exp/VQA_E_VCR/
export CUDNN_INCLUDE_DIR=/usr/local/cuda-10.0/include/
export CUDNN_LIB_DIR=/usr/local/cuda-10.0/lib64/
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-10.0/

# nohup python train.py -params multiatt/default.json -folder saves/flagship_answer > ../out/train_q2a_log.txt 2>&1 &

# nohup python train.py -params multiatt/default.json -folder saves/flagship_answer -rationale > ../out/train_qa2r_log.txt 2>&1 &

# tail -f ../out/train_qa2r_log.txt


# python eval_q2ar.py -answer_preds saves/flagship_answer/valpreds.npy -rationale_preds saves/flagship_rationale/valpreds.npy > ../out/eval_log.txt 2>&1 &

# tail -f ../out/eval_log.txt 
