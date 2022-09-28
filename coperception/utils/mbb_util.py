import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

from coperception.datasets import V2XSimDet, MbbSampler
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import *
from coperception.utils.loss import *
from coperception.models.det import *
from coperception.utils import AverageMeter
from coperception.utils.data_util import apply_pose_noise
from coperception.utils.mean_ap import eval_map, get_residual_error_and_cov

import glob
import os

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def test_model(fafmodule, validation_data_loader, flag, device, config, epoch, args):
    fafmodule.model.eval()
    num_agent = args.num_agent
    apply_late_fusion = args.apply_late_fusion
    agent_idx_range = range(num_agent) if args.rsu else range(1, num_agent)
    save_epoch_path = check_folder(os.path.join(args.test_store, str(epoch)))
    save_fig_path = [
        check_folder(os.path.join(save_epoch_path, f"vis{i}")) for i in agent_idx_range
    ]
    tracking_path = [
        check_folder(os.path.join(save_epoch_path, f"result{i}"))
        for i in agent_idx_range
    ]

    # for local and global mAP evaluation
    det_results_local = [[] for i in agent_idx_range]
    annotations_local = [[] for i in agent_idx_range]

    if not args.rsu:
        num_agent -= 1
    tracking_file = [set()] * num_agent
    
    for cnt, sample in enumerate(validation_data_loader):
        t = time.time()
        (
            padded_voxel_point_list,
            padded_voxel_points_teacher_list,
            label_one_hot_list,
            reg_target_list,
            reg_loss_mask_list,
            anchors_map_list,
            vis_maps_list,
            gt_max_iou,
            filenames,
            target_agent_id_list,
            num_agent_list,
            trans_matrices_list,
        ) = zip(*sample)

        print(filenames)

        filename0 = filenames[0]
        trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
        target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
        num_all_agents = torch.stack(tuple(num_agent_list), 1)

        # add pose noise
        if args.pose_noise > 0:
            apply_pose_noise(args.pose_noise, trans_matrices)

        if not args.rsu:
            num_all_agents -= 1

        if flag == "upperbound":
            padded_voxel_points = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
        else:
            padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)

        label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
        reg_target = torch.cat(tuple(reg_target_list), 0)
        reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
        anchors_map = torch.cat(tuple(anchors_map_list), 0)
        vis_maps = torch.cat(tuple(vis_maps_list), 0)

        data = {
            "bev_seq": padded_voxel_points.to(device),
            "labels": label_one_hot.to(device),
            "reg_targets": reg_target.to(device),
            "anchors": anchors_map.to(device),
            "vis_maps": vis_maps.to(device),
            "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
            "target_agent_ids": target_agent_ids.to(device),
            "num_agent": num_all_agents.to(device),
            "trans_matrices": trans_matrices.to(device),
        }

        if flag == "lowerbound_box_com":
            loss, cls_loss, loc_loss, result = fafmodule.predict_all_with_box_com(
                data, data["trans_matrices"]
            )
        elif flag == "disco":
            (
                loss,
                cls_loss,
                loc_loss,
                result,
                save_agent_weight_list,
            ) = fafmodule.predict_all(data, 1, num_agent=num_agent)
        else:
            loss, cls_loss, loc_loss, result = fafmodule.predict_all(
                data, 1, num_agent=num_agent
            )

        box_color_map = ["red", "yellow", "blue", "purple", "black", "orange"]

        # If has RSU, do not count RSU's output into evaluation
        eval_start_idx = 1 if args.rsu else 0

        # local qualitative evaluation
        for k in range(eval_start_idx, num_agent):
            box_colors = None
            if apply_late_fusion == 1 and len(result[k]) != 0:
                pred_restore = result[k][0][0][0]["pred"]
                score_restore = result[k][0][0][0]["score"]
                selected_idx_restore = result[k][0][0][0]["selected_idx"]

            data_agents = {
                "bev_seq": torch.unsqueeze(padded_voxel_points[k, :, :, :, :], 1),
                "reg_targets": torch.unsqueeze(reg_target[k, :, :, :, :, :], 0),
                "anchors": torch.unsqueeze(anchors_map[k, :, :, :, :], 0),
            }
            temp = gt_max_iou[k]

            if len(temp[0]["gt_box"]) == 0:
                data_agents["gt_max_iou"] = []
            else:
                data_agents["gt_max_iou"] = temp[0]["gt_box"][0, :, :]

            # late fusion
            if apply_late_fusion == 1 and len(result[k]) != 0:
                box_colors = late_fusion(
                    k, num_agent, result, trans_matrices, box_color_map
                )

            result_temp = result[k]

            temp = {
                "bev_seq": data_agents["bev_seq"][0, -1].cpu().numpy(),
                "result": [] if len(result_temp) == 0 else result_temp[0][0],
                "reg_targets": data_agents["reg_targets"].cpu().numpy()[0],
                "anchors_map": data_agents["anchors"].cpu().numpy()[0],
                "gt_max_iou": data_agents["gt_max_iou"],
            }
            det_results_local[k], annotations_local[k], det_results_frame, annotations_frame = cal_local_mAP(
                config, temp, det_results_local[k], annotations_local[k], True
            )

            filename = str(filename0[0][0])
            cut = filename[filename.rfind("agent") + 7 :]
            seq_name = cut[: cut.rfind("_")]
            idx = cut[cut.rfind("_") + 1 : cut.rfind("/")]
            seq_save = os.path.join(save_fig_path[k], seq_name)
            check_folder(seq_save)
            idx_save = str(idx) + ".png"
            #temp_ = deepcopy(temp)
            if args.visualization:
                visualization(
                    config,
                    temp,
                    box_colors,
                    box_color_map,
                    apply_late_fusion,
                    os.path.join(seq_save, idx_save),
                )

            # restore data before late-fusion
            if apply_late_fusion == 1 and len(result[k]) != 0:
                result[k][0][0][0]["pred"] = pred_restore
                result[k][0][0][0]["score"] = score_restore
                result[k][0][0][0]["selected_idx"] = selected_idx_restore

        print("Validation scene {}, at frame {}".format(seq_name, idx))
        print("Takes {} s\n".format(str(time.time() - t)))

    log_file_path = os.path.join(args.test_store, "log_test.txt")
    if os.path.exists(log_file_path):
        log_file = open(log_file_path, "a")
    else:
        log_file = open(log_file_path, "w")

    def print_and_write_log(log_str):
        print(log_str)
        log_file.write(log_str + "\n")

    # local mAP evaluation
    det_results_all_local = []
    annotations_all_local = []
    mean_ap_5 = []
    mean_ap_7 = []
    mean_ap_all = []
    for k in range(eval_start_idx, num_agent):
        if type(det_results_local[k]) != list or len(det_results_local[k]) == 0:
            continue

        print_and_write_log("Local mAP@0.5 from agent {}".format(k))
        mean_ap, _ = eval_map(
            det_results_local[k],
            annotations_local[k],
            scale_ranges=None,
            iou_thr=0.5,
            dataset=None,
            logger=None,
        )
        mean_ap_5.append(mean_ap)
        print_and_write_log("Local mAP@0.7 from agent {}".format(k))

        mean_ap, _ = eval_map(
            det_results_local[k],
            annotations_local[k],
            scale_ranges=None,
            iou_thr=0.7,
            dataset=None,
            logger=None,
        )
        mean_ap_7.append(mean_ap)

        det_results_all_local += det_results_local[k]
        annotations_all_local += annotations_local[k]

    npy_frame_file = os.path.join(save_epoch_path, "all_data.npy")
    det_res = {"det_results_frame": det_results_all_local, "annotations_frame": annotations_all_local}
    np.save(npy_frame_file, det_res)
    mean_ap_local_average, _ = eval_map(
        det_results_all_local,
        annotations_all_local,
        scale_ranges=None,
        iou_thr=0.5,
        dataset=None,
        logger=None,
    )
    mean_ap_all.append(mean_ap_local_average)

    mean_ap_local_average, _ = eval_map(
        det_results_all_local,
        annotations_all_local,
        scale_ranges=None,
        iou_thr=0.7,
        dataset=None,
        logger=None,
    )
    mean_ap_all.append(mean_ap_local_average)
    mean_ap_agents = []
    mean_ap_agents.append(mean_ap_5)
    mean_ap_agents.append(mean_ap_7)

    print_and_write_log(
        "Quantitative evaluation results of model, at epoch {}".format(
             epoch
        )
    )

    print_and_write_log(
        "average local mAP@0.5 is {} and average local mAP@0.7 is {}".format(
            mean_ap_all[0], mean_ap_all[1]
        )
    )

def computer_mbb_covar(args):
    start_epoch = 0
    end_epoch = args.nepoch
    res_diff = []
    all_predicted_covariance = []
    covar_flag = False
    iou_thr = 0.5
    for epoch in range(start_epoch, end_epoch+1):
        data_path = args.test_store + "/{}".format(epoch) +"/all_data.npy"
        print("Load data from {}".format(data_path))
        data = np.load(data_path, allow_pickle=True)
        det_results_all_local = data.item()['det_results_frame']
        annotations_all_local = data.item()['annotations_frame']
        res_diff_one_epoch, predicted_covar = get_residual_error_and_cov(det_results_all_local, annotations_all_local, scale_ranges=None, iou_thr=iou_thr)
        res_diff.extend(res_diff_one_epoch)
        if predicted_covar != None:
            all_predicted_covariance.extend(predicted_covar)
            covar_flag = True
        print("Number of corners of all bounding box: {}".format(len(res_diff[epoch])))
    res_diff_np = np.array(res_diff[0])
    if covar_flag:
        all_predicted_covariance_np = np.array(all_predicted_covariance[0])
    for i in range(1, len(res_diff)):
        res_diff_np = np.concatenate((res_diff_np, res_diff[i]))
        if covar_flag:
            all_predicted_covariance_np = np.concatenate((all_predicted_covariance_np, all_predicted_covariance[i]))
    print(res_diff_np.shape)
    print("covariance matrix for residual error:")
    covar_e = np.cov(res_diff_np.T)
    print(covar_e)
    save_data = {"covar_e":covar_e}
    if covar_flag:
        print(all_predicted_covariance_np.shape)
        print("mean of predicted covariance matrix:")
        covar_a = np.mean(all_predicted_covariance_np, axis=0)
        print(covar_a)
        save_data['covar_a'] =  covar_a
    save_data_path = args.test_store + "/mbb_covar.npy"
    np.save(save_data_path, save_data)
    print("Save computed covariance in {}".format(save_data_path))