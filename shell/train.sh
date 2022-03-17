#train
#CUDA_VISIBLE_DEVICES=1 python -m manydepth.train --data_path /root/datasets/kitti_data/ --log_dir /root/ManyDepth/results/ --batch_size 8 --model_name original_rmake
#CUDA_VISIBLE_DEVICES=1 python -m manydepth.train --data_path /root/datasets/kitti_data/ --log_dir /root/ManyDepth/results/ --batch_size 12 --using_probability_depth --model_name change_lowest_lost_inv
#CUDA_VISIBLE_DEVICES=2 python -m manydepth.train --data_path /root/datasets/kitti_data/ --log_dir /root/ManyDepth/results/ --batch_size 12 --using_GTPose --using_scale_factor --model_name gps_scaletor103
#train with disable_motion_masking
#CUDA_VISIBLE_DEVICES=2 python -m manydepth.train --data_path /root/datasets/kitti_data/ --log_dir /root/ManyDepth/results/ --batch_size 12 --num_epochs 25 --freeze_teacher_epoch 20 --using_probability_depth --using_GTPose --model_name adjust_gps_weight_probability
#train with no_matching_augmentation
#CUDA_VISIBLE_DEVICES=1 python -m manydepth.train --data_path /root/datasets/kitti_data/ --log_dir /root/ManyDepth/results/ --batch_size 8 --no_matching_augmentation --using_probability_depth --using_GTPose --model_name original_gps_constant104
#evaluate depth
#CUDA_VISIBLE_DEVICES=2 python -m manydepth.evaluate_depth --data_path /root/datasets/kitti_data/ --load_weights_folder /root/ManyDepth/results/constraint_posenet_translation_105/models/weights_19/ --eval_mono
CUDA_VISIBLE_DEVICES=2 python -m manydepth.evaluate_depth --data_path /root/datasets/kitti_data/ --load_weights_folder /root/ManyDepth/results/gps_scaletor103/models/weights_19/ --using_scale_factor --eval_mono