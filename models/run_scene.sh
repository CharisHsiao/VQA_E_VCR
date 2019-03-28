# source actiavte nsvqa
export PYTHONPATH=/home/lab/A305/yifan/vqa/vqa-exp/VQA_E_VCR/
export CUDNN_INCLUDE_DIR=/usr/local/cuda-10.0/include/
export CUDNN_LIB_DIR=/usr/local/cuda-10.0/lib64/
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-10.0/

nohup python train.py -params multiatt_scene/default.json -folder saves/multiatt_scene/flagship_answer > ../out/scene_train_q2a_log.txt 2>&1 &
tail -f ../out/scene_train_q2a_log.txt 
# nohup python train.py -params multiatt_scene/default.json -folder saves/multiatt_scene/flagship_answer -rationale > ../out/scene_train_qa2r_log.txt 2>&1 &

# python train.py -params multiatt_scene/default.json -folder saves/mulitt_scene/flagship_answer

# tail -f ../out/train_qa2r_log.txt


# python eval_q2ar.py -answer_preds saves/flagship_answer/valpreds.npy -rationale_preds saves/flagship_rationale/valpreds.npy > ../out/eval_log.txt 2>&1 &

# tail -f ../out/eval_log.txt 
