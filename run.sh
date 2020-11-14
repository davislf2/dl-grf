### This is a script to put all models to train 

# FCN
time CUDA_VISIBLE_DEVICES=0 python3 main.py \
--dir ./data \
--action single \
--archive pretrain \
--dataset one_shot_hc \
--classifier fcn \
--file_ext .npy \
--itr _itr_8 \
--trainable_layers [10,11] \
--nb_epochs 10 \
--nb_epochs_finetune 100

# ResNet
time CUDA_VISIBLE_DEVICES=0 python3 main.py \
--dir ./data \
--action single \
--archive pretrain \
--dataset one_shot_hc \
--classifier resnet \
--file_ext .npy \
--itr _itr_8 \
--trainable_layers [36,37] \
--nb_epochs 10 \
--nb_epochs_finetune 100

# CNN
time CUDA_VISIBLE_DEVICES=0 python3 main.py \
--dir ./data \
--action single \
--archive pretrain \
--dataset one_shot_hc \
--classifier cnn \
--file_ext .npy \
--itr _itr_8 \
--trainable_layers [5,6] \
--nb_epochs 10 \
--nb_epochs_finetune 100

# MLP
time CUDA_VISIBLE_DEVICES=0 python3 main.py \
--dir ./data \
--action single \
--archive pretrain \
--dataset one_shot_hc \
--classifier mlp \
--file_ext .npy \
--itr _itr_8 \
--trainable_layers [8,9] \
--nb_epochs 10 \
--nb_epochs_finetune 100