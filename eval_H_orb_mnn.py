import numpy as np
import cv2
import torch
import os
import os.path as osp
import argparse
import json
import sys
from pathlib import Path
from time import time
from tqdm import tqdm
import kornia


orb_path = Path(__file__).parent / "orb/lib"
sys.path.append(str(orb_path))
from orb_slam2_extractor import ORBExtractor


def mnn_matcher(descriptors0, descriptors1, topk=-1):
    descriptors0 = descriptors0.astype(np.float32)
    descriptors1 = descriptors1.astype(np.float32)
    desc0_ts = torch.from_numpy(descriptors0)
    desc1_ts = torch.from_numpy(descriptors1)
    dm = torch.cdist(desc0_ts, desc1_ts)
    desc_dist_ts, matches_ts = kornia.kornia.feature.match_mnn(desc0_ts, desc1_ts, dm)
    desc_dist = desc_dist_ts.detach().cpu().numpy()
    matches = matches_ts.detach().cpu().numpy()

    min_ids_dist = np.argsort(desc_dist, axis=0).squeeze(1)[:topk]
    topk_desc_dist = desc_dist[:topk]
    topk_matches = matches[min_ids_dist]
    return topk_matches, topk_desc_dist


def extract_featrues(gray_image, extractor):
    features = extractor.ExtarctFeatures(gray_image)
    kpts_tuples, descriptors = features
    keypoints_cv = [cv2.KeyPoint(*kp) for kp in kpts_tuples]
    # TODO: kpt.size / 31 means ?
    keypoints = np.array(
        [[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] for kp in keypoints_cv], 
        dtype=np.float32
    )
    unnorm_kpts = keypoints.copy()
    return keypoints, descriptors, unnorm_kpts


def error_auc(errors, thresholds=[3, 5, 10]):
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)
    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}, aucs


def adapt_homography_to_processing(H, new_shape, ori_shape0, ori_shape1):
    new_shape = np.array(new_shape)
    ori_shape0 = np.array(ori_shape0)
    ori_shape1 = np.array(ori_shape1)

    scale0 = max(new_shape / ori_shape0)
    up_scale = np.diag(np.array([1. / scale0, 1. / scale0, 1.]))

    scale1 = max(new_shape / ori_shape1)
    down_scale = np.diag(np.array([scale1, scale1, 1.]))

    H = down_scale @ H @ up_scale
    return H


def get_bitmap(image_path, new_shape=None):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ori_shape = image.shape[:2]

    if new_shape:
        new_shape = np.array(new_shape)
        ori_shape = np.array(ori_shape)
        scale = max(new_shape / ori_shape)

        image = image[:int(new_shape[0] / scale), 
                      :int(new_shape[1] / scale)]
        image = cv2.resize(
            image, (new_shape[1], new_shape[0]),
            interpolation=cv2.INTER_AREA)
    return image, ori_shape


def eval_hpatches(args):
    data_path = args.hpatches_path
    resize = args.resize
    scenes = sorted(os.listdir(data_path))
    i_results = []
    v_results = []
    mean_of_inliers = 0
    extractor = ORBExtractor(2048, 1.2, 8)
    
    start_time = time()
    for scene in tqdm(scenes[::-1], total=len(scenes)):
        scene_cls = scene.split('_')[0]
        scene_path = os.path.join(data_path, scene)

        im_p0 = os.path.join(scene_path, '1.ppm')
        im0, ori_shape0 = get_bitmap(im_p0, resize)

        # Compute refernece image local features
        keypoints0, descriptors0, unnorm_kpts0 = extract_featrues(im0, extractor)
    
        shape = im0.shape[:2]
        corners = np.array([
            [0,            0,            1],
            [shape[1] - 1, 0,            1],
            [0,            shape[0] - 1, 1],
            [shape[1] - 1, shape[0] - 1, 1]
        ])
        sum_of_inliers = 0
        for idx in range(2, 7):
            im_p1 = os.path.join(scene_path, f'{idx}.ppm')
            im1, ori_shape1 = get_bitmap(im_p1, resize)
            
            # Compute target image feature
            keypoints1, descriptors1, unnorm_kpts1 = extract_featrues(im1, extractor)

            # Compute matches using MNN
            pts0 = unnorm_kpts0[:, :2]
            pts1 = unnorm_kpts1[:, :2]
            
            # Mutual nearest neighbors (MNN)
            # shape: [nums, 2] (idx0, idx1)
            matches, _ = mnn_matcher(descriptors0, descriptors1, topk=args.topk_match)

            mids0 = matches[:, 0]
            mids1 = matches[:, 1]
            match_keypoints0 = pts0[mids0]
            match_keypoints1 = pts1[mids1]

            
            if len(match_keypoints0) < 4:
                failed_error = 999
                if scene_cls == 'i':
                    i_results.append(failed_error)
                else:
                    v_results.append(failed_error)
                continue

            H_pred, mask = cv2.findHomography(
                match_keypoints0, match_keypoints1, method=cv2.RANSAC
            )
            if H_pred is None:
                failed_error = 999
                if scene_cls == 'i':
                    i_results.append(failed_error)
                else:
                    v_results.append(failed_error)
                continue

            mask = mask.flatten().astype(bool)
            n_inliers = np.sum(mask)
            sum_of_inliers += float(n_inliers)

            # Pred corners
            pred_corners = np.dot(corners, np.transpose(H_pred))
            pred_corners = pred_corners[:, :2] / pred_corners[:, 2:]

            H_real = np.loadtxt(osp.join(scene_path, f'H_1_{idx}'))
            H_real = adapt_homography_to_processing(H_real, resize, ori_shape0, ori_shape1)
            # Real corners
            real_corners = np.dot(corners, np.transpose(H_real))
            real_corners = real_corners[:, :2] / real_corners[:, 2:]

            error = np.mean(np.linalg.norm(real_corners - pred_corners, axis=1))

            if scene_cls == 'i':
                i_results.append(error)
            else:
                v_results.append(error)

        mean_of_inliers += sum_of_inliers / 5.
        total_time = time() - start_time
    
    mean_of_inliers /= float(len(scenes))
    v_results = np.array(v_results).astype(np.float32)
    i_results = np.array(i_results).astype(np.float32)
    results = np.concatenate((i_results, v_results), axis=0)

    # Compute auc
    auc_of_homo_i, aucs_i = error_auc(i_results, thresholds=args.auc_threshold)
    auc_of_homo_v, aucs_v = error_auc(v_results, thresholds=args.auc_threshold)
    auc_of_homo, aucs = error_auc(results, thresholds=args.auc_threshold)

    dumps = {
        "times": total_time,
        "inliers": mean_of_inliers,
        **{k: v * 100. for k, v in auc_of_homo.items()},
        **{f"i_{k}": v * 100. for k, v in auc_of_homo_i.items()},
        **{f"v_{k}": v * 100. for k, v in auc_of_homo_v.items()},
    }
    print(f'Test in HPatches, Homography results: \n{dumps}')
    return dumps, aucs, aucs_i, aucs_v


def main(args):
    dumps, aucs, aucs_i, aucs_v = eval_hpatches(args)
    dump_name = f'orb_mnn_topK_{args.topk_match}_{args.auc_threshold}.json'
    dumps_save_path = osp.join(args.json_save_root, dump_name) 
    with open(dumps_save_path, 'w') as f:
        f.write(json.dumps(dumps, indent=4))
    np.save('eval_dumps/npy/orb_mnn_auc_i.npy', aucs_i)
    np.save('eval_dumps/npy/orb_mnn_auc_v.npy', aucs_v)
    np.save('eval_dumps/npy/orb_mnn_aucs.npy', aucs)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -------------------------------- HPATCHES IMAGE -------------------------------- #
    parser.add_argument('--hpatches_path', default='/home/xiaohunan/WorkSpace/Datasets/hpatches-sequences-release/')
    parser.add_argument('--json_save_root', default='eval_dumps/homography/')
    parser.add_argument('--resize', default=(480, 640))
    parser.add_argument('--topk_match', default=100)
    parser.add_argument('--auc_threshold', default=[1, 3, 5])
    args = parser.parse_args()
    main(args)