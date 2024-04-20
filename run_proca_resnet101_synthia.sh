########### ResNet-101 synthia -> cityscapes
NGPUS=1
# train on depth data
# torchrun --nproc_per_node=$NGPUS train_depth.py -cfg configs/synthia/deeplabv2_r101_src.yaml SOLVER.LAMBDA_LOV 0.75 OUTPUT_DIR results/synthia/r101_g2c_depth SOLVER.BATCH_SIZE 4

# train on source data
#  torchrun --nproc_per_node=$NGPUS train_src.py -cfg configs/synthia/deeplabv2_r101_src.yaml SOLVER.LAMBDA_LOV 0.75 OUTPUT_DIR results/synthia/r101_g2c_src SOLVER.BATCH_SIZE 1

# python3 prototype_dist_init.py -cfg configs/synthia/deeplabv2_r101_src.yaml resume results/synthia/r101_g2c_src/model_best.pth OUTPUT_DIR results/synthia/r101_g2c_src/ SOLVER.BATCH_SIZE_VAL 1

# train proca
# python3 train_proca.py -cfg configs/synthia/deeplabv2_r101_proca.yaml resume results/synthia/r101_g2c_src/model_best.pth CV_DIR results/synthia/r101_g2c_src/ OUTPUT_DIR results/synthia/r101_g2c_ours_proca/ SOLVER.BATCH_SIZE 2

# Generate outputs for all proca classifiers ---- CHANGE TARGET_TRAIN to "cityscapes_val" in YAML FILE!!!! ----
# python3 Proca_outputs.py -cfg configs/synthia/deeplabv2_r101_pseudo.yaml DATASETS.TEST cityscapes_train resume results/synthia/Generate_Decoder_labels/model_best.pth OUTPUT_DIR results/synthia/Generate_Decoder_labels SOLVER.BATCH_SIZE_VAL 1

# generate pseudo label
# python3 DACS.py -cfg configs/synthia/deeplabv2_r101_pseudo.yaml DATASETS.TEST cityscapes_train resume results/synthia/r101_g2c_ours_proca/model_best.pth OUTPUT_DIR results/synthia/r101_g2c_ours_proca SOLVER.BATCH_SIZE_VAL 1

# train with generated pseudo label
# torchrun --nproc_per_node=$NGPUS train_dassl.py -cfg configs/synthia/deeplabv2_r101_ssl.yaml resume results/synthia/r101_g2c_ours_proca_ssl/model_current.pth PREPARE_DIR results/synthia/r101_g2c_ours_proca/ OUTPUT_DIR results/synthia/r101_g2c_ours_proca_ssl SOLVER.BATCH_SIZE 2

# generate pseudo label
# python3 pseudo_label_proda.py -cfg configs/synthia/deeplabv2_r101_ssl.yaml DATASETS.TEST cityscapes_train resume results/synthia/r101_g2c_ours_proca_ssl/model_best.pth OUTPUT_DIR results/synthia/r101_g2c_ours_proca_ssl SOLVER.BATCH_SIZE_VAL 1
# 
# train with generated pseudo label
# torchrun --nproc_per_node=$NGPUS train_ssl.py -cfg configs/synthia/deeplabv2_r101_ssl.yaml resume results/synthia/r101_g2c_final/model_current.pth PREPARE_DIR results/synthia/r101_g2c_ours_proca_ssl/ OUTPUT_DIR results/synthia/r101_g2c_final SOLVER.BATCH_SIZE 2

torchrun --nproc_per_node=$NGPUS eval_ssl.py -cfg configs/synthia/deeplabv2_r101_proca.yaml resume results/synthia/Generate_Decoder_labels/model_best.pth PREPARE_DIR results/synthia/Generate_Decoder_labels/ OUTPUT_DIR results/synthia/Generate_Decoder_labels/ SOLVER.BATCH_SIZE 2
