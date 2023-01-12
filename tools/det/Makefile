# Path to the original V2X-Sim dataset
original_data_path := /data/v2x-sim
# Where to save the created data
#V2X-Sim-det  V2X-Sim-demo
create_data_save_path := /data/v2x-sim-nyu/V2X-Sim-det
# Index of the begining scene
scene_begin := 0
 # Index of the ending scene + 1
scene_end := 100
# Index of the start agent
from_agent := 0
# Index of the end agent + 1
to_agent := 6

training_script := train_codet.py
# Path to the created training data
training_data := $(create_data_save_path)/train

testing_script := test_codet.py
# Path to the test/val data
testing_data := $(create_data_save_path)/test
val_data := $(create_data_save_path)/val
# [lowerbound / upperbound / v2v / disco / when2com / max / mean / sum / agent]
com := disco
batch_size := 1
# [train / test / val]
split := train
# Where to store the logs
logpath := check/check_loss_base
testlogpath := check/test_loss_corner_pair_ind
# Train for how many epochs
nepoch := 25
# loss type: corner_loss faf_loss kl_loss_center kl_loss_center_add, kl_loss_corner, kl_loss_center_ind, kl_loss_center_offset_ind, kl_loss_corner_pair_ind 
loss_type = kl_loss_corner_pair_ind
# If given, the model will resume from its most recent (by modification time) check point
auto_resume_path := $(logpath)
# experiment name
exp_name = test_mbb_base
# compress_level
compress_level := 0
# 1: apply late fusion. 0: no late fusion
apply_late_fusion := 0
# 1: do visualizaton. 0: no visualization
visualization := 0
# pose_noise in meters
pose_noise := 0
# only apply v2i communication
only_v2i := 0
# 0: no RSU, 1: RSU
rsu := 0
# Whether to use pose info for When2com
warp_flag := 0
# Used when testing when2com / who2com
# when2com: activated, who2com: argmax_test
inference := activated
#block len for MBB method
block_len := 16

create_data:
	python create_data_det.py \
	--root $(original_data_path) \
	--split $(split) \
	--scene_begin $(scene_begin) \
	--scene_end $(scene_end) \
	--savepath $(create_data_save_path) \
	--from_agent $(from_agent) \
	--to_agent $(to_agent)

train:
	python $(training_script) \
	--data $(training_data) \
	--com $(com) \
	--log \
	--batch_size $(batch_size) \
	--auto_resume_path $(auto_resume_path) \
	--nepoch $(nepoch) \
	--logpath $(logpath) \
	--warp_flag $(warp_flag) \
	--rsu $(rsu) \
	--compress_level $(compress_level) \
	--pose_noise $(pose_noise) \
	--only_v2i $(only_v2i) \
	--loss_type $(loss_type) \
	--exp_name $(exp_name)


train_disco:
	python $(training_script) \
	--data $(training_data) \
	--com disco \
	--log --batch $(batch_size) \
	--kd_flag 1 \
	--resume_teacher $(logpath)/upperbound/with_rsu/epoch_$(nepoch).pth \
	--auto_resume_path $(auto_resume_path) \
	--logpath $(logpath) \
	--nepoch $(nepoch) \
	--compress_level $(compress_level) \
	--pose_noise $(pose_noise) \
	--only_v2i $(only_v2i) \
	-- rsu 1 \
	--loss_type $(loss_type)

train_disco_no_rsu:
	python $(training_script) \
	--data $(training_data) \
	--com disco \
	--log --batch $(batch_size) \
	--kd_flag 1 \
	--resume_teacher check/check_loss_base/upperbound/no_rsu/epoch_100.pth \
	--auto_resume_path $(auto_resume_path) \
	--logpath $(logpath) \
	--nepoch $(nepoch) \
	--compress_level $(compress_level) \
	--pose_noise $(pose_noise) \
	--only_v2i $(only_v2i) \
	--rsu 0 \
	--loss_type $(loss_type) \
	--exp_name $(exp_name)
    
mbb_train:
	python train_mbb.py \
	--data $(training_data) \
	--com $(com) \
	--log \
	--batch_size $(batch_size) \
	--auto_resume_path $(auto_resume_path) \
	--nepoch $(nepoch) \
	--logpath $(logpath) \
	--warp_flag $(warp_flag) \
	--rsu $(rsu) \
	--compress_level $(compress_level) \
	--pose_noise $(pose_noise) \
	--only_v2i $(only_v2i) \
	--block_len $(block_len) \
	--init_resume_path check/check_loss_base/$(com)/no_rsu/epoch_80.pth \
	--test_store $(logpath)/$(com)/no_rsu \
	--test_data $(create_data_save_path) \
	--loss_type $(loss_type) \
	--exp_name $(exp_name) \
	--test
    
mbb_train_disco_no_rsu:
	python train_mbb.py \
	--data $(training_data) \
	--com disco \
	--log --batch $(batch_size) \
	--kd_flag 1 \
	--resume_teacher check/check_loss_base/upperbound/no_rsu/epoch_100.pth \
	--auto_resume_path $(auto_resume_path) \
	--logpath $(logpath) \
	--nepoch $(nepoch) \
	--compress_level $(compress_level) \
	--pose_noise $(pose_noise) \
	--only_v2i $(only_v2i) \
	--rsu 0 \
	--block_len $(block_len) \
	--init_resume_path check/check_loss_base/disco/no_rsu/epoch_80.pth \
	--test_store $(logpath)/disco/no_rsu \
	--test_data $(create_data_save_path) \
	--loss_type $(loss_type) \
	--exp_name $(exp_name) \
	--test
    
test:
	python $(testing_script) \
	--data $(testing_data) \
	--com $(com) \
	--resume $(logpath)/$(com)/with_rsu/epoch_$(nepoch).pth \
	--tracking \
	--logpath $(logpath) \
	--apply_late_fusion $(apply_late_fusion) \
	--visualization $(visualization) \
	--inference $(inference) \
	--warp_flag $(warp_flag) \
	--rsu 1 \
	--covar_path $(logpath)/${com}/no_rsu/mbb_covar.npy \
	--loss_type $(loss_type)

test_no_rsu:
	python $(testing_script) \
	--data $(testing_data) \
	--com $(com) \
	--resume $(logpath)/$(com)/no_rsu/epoch_$(nepoch).pth \
	--logpath $(logpath) \
	--apply_late_fusion $(apply_late_fusion) \
	--visualization $(visualization) \
	--inference $(inference) \
	--warp_flag $(warp_flag) \
	--rsu 0 \
	--covar_path $(logpath)/${com}/no_rsu/mbb_covar.npy \
	--loss_type $(loss_type)

mbb_test_no_rsu:
	python test_mbb.py \
	--data $(val_data) \
	--com $(com) \
	--resume $(logpath)/$(com)/no_rsu \
	--logpath $(logpath)/${com}/no_rsu \
	--apply_late_fusion $(apply_late_fusion) \
	--visualization $(visualization) \
	--inference $(inference) \
	--warp_flag $(warp_flag) \
	--rsu 0 \
	--test_store $(logpath)/$(com)/no_rsu \
	--init_resume_path check/check_loss_base/$(com)/no_rsu/epoch_80.pth \
	--nepoch ${nepoch} \
	--loss_type $(loss_type)

compute_mbb_covar:
	python compute_mbb_covar.py \
	--mbb_path $(logpath)/${com}/no_rsu \
	--min_epoch 0 \
	--max_epoch $(nepoch)