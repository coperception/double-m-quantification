import argparse
import os
from copy import deepcopy

from coperception.utils.CoDetModule import *
from coperception.utils.loss import *
from coperception.utils.mean_ap import eval_map, eval_nll, get_residual_error_and_cov

def main(args):
    start_epoch = args.min_epoch
    end_epoch = args.max_epoch
    res_diff = []
    all_predicted_covariance = []
    covar_flag = False
    iou_thr = 0.5
    for epoch in range(start_epoch, end_epoch+1):
        data_path = args.mbb_path + "/{}".format(epoch) +"/all_data.npy"
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
    save_data_path = args.mbb_path + "/mbb_covar.npy"
    np.save(save_data_path, save_data)
    print("Save computed covariance in {}".format(save_data_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_epoch", default=0, type=int, help="min epochs we consider")
    parser.add_argument("--max_epoch", default=25, type=int, help="max epochs we consider")
    parser.add_argument("--nworker", default=1, type=int, help="Number of workers")
    parser.add_argument(
        "--mbb_path",
        default="",
        type=str,
        help="The path to the serval mbb models",
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    main(args)