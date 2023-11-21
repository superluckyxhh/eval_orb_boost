import numpy as np
import torch
import cv2
import sys
import os
import os.path as osp
import yaml
import argparse
from pathlib import Path
import kornia
from models.featurebooster import FeatureBooster

orb_path = Path(__file__).parent / "orb/lib"
sys.path.append(str(orb_path))
from orb_slam2_extractor import ORBExtractor


def read_image(image_path, new_shape=None):
    image_rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)
    origin_shape = image_rgb.shape[:2]

    if new_shape:
        new_shape = np.array(new_shape)
        origin_shape = np.array(origin_shape)
        scale = max(new_shape / origin_shape)
        image_rgb = image_rgb[:int(new_shape[0] / scale), :int(new_shape[1] / scale), :]
        image_rgb = cv2.resize(image_rgb, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)

    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    return image_rgb, gray_image, origin_shape

def mnn_matcher(descriptors0, descriptors1, topk=-1):
    descriptors0 = descriptors0.astype(np.float32)
    descriptors1 = descriptors1.astype(np.float32)
    desc0_ts = torch.from_numpy(descriptors0)
    desc1_ts = torch.from_numpy(descriptors1)
    dm = torch.cdist(desc0_ts, desc1_ts)
    desc_dist_ts, matches_ts = kornia.kornia.feature.match_mnn(desc0_ts, desc1_ts, dm)
    desc_dist = desc_dist_ts.detach().cpu().numpy()
    matches = matches_ts.detach().cpu().numpy()

    min_ids_dist = np.argsort(desc_dist, axis=0).squeeze(1)
    min_ids_dist_topk = min_ids_dist[:topk]
    topk_desc_dist = desc_dist[min_ids_dist_topk]
    topk_matches = matches[min_ids_dist_topk]
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
   

def plot_matching(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                text, save_path, show_all_kpts=False, margin=5):
    H0, W0, _ = image0.shape
    H1, W1, _ = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0+margin:, :] = image1

    if show_all_kpts:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    # Show matching kpts
    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    red_color = (0, 0, 255)
    green_color = (0, 255, 0)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=green_color, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, red_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, red_color, -1,
                   lineType=cv2.LINE_AA)
    
    sc = min(H / 640., 2.0)
    Ht = int(25 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    0.7*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    0.7*sc, txt_color_fg, 1, cv2.LINE_AA)
    cv2.imwrite(save_path, out)
   

def main(args):
    # Visual Odeometry: ORB extractor
    extractor = ORBExtractor(2048, 1.2, 8)
    
    # Set scene ID
    scene_path = osp.join(args.hpatches_root, args.scene_id)
    
    # Read 1st frame & extract features
    image0_path = osp.join(scene_path, '1.ppm')
    image0, gray0, origin_shape0 = read_image(image0_path, args.resize)
    keypoints0, descriptors0, unnorm_kpts0 = extract_featrues(gray0, extractor)
    
    # Read continuous maps
    for idx in range(2, 7):
        image1_path = osp.join(scene_path, f'{idx}.ppm')
        image1, gray1, origin_shape1 = read_image(image1_path, args.resize)
        keypoints1, descriptors1, unnorm_kpts1 = extract_featrues(gray1, extractor)
       
        # Only x,y coord
        pts0 = unnorm_kpts0[:, :2]
        pts1 = unnorm_kpts1[:, :2]
        
        # Mutual nearest neighbors (MNN)
        # shape: [nums, 2] (idx0, idx1)
        matches, _ = mnn_matcher(descriptors0, descriptors1, topk=args.topk_match)

        mids0 = matches[:, 0]
        mids1 = matches[:, 1]
        mkpts0 = pts0[mids0]
        mkpts1 = pts1[mids1]

        # Plot matching
        if args.plot_flag:
            text = ['ORB - MNN',
                    f'Image: {args.scene_id[:-1]}/1-{idx}',
                    f'Topk matches: {args.topk_match}']
            save_dir = args.plot_path + args.scene_id
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            save_path = save_dir + f'orb_mnn_1_{idx}.png'
            plot_matching(image0, image1, keypoints0, keypoints1, mkpts0, mkpts1, text, save_path, show_all_kpts=False)
            print(f'Plot matching .png save in {save_path}')  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hpatches_root', default='/home/xiaohunan/WorkSpace/Datasets/hpatches-sequences-release/')
    parser.add_argument('--scene_id', default='i_ajuntament/', choices=['v_there/', 'i_ajuntament/'])
    parser.add_argument('--resize', default=[480, 640])
    parser.add_argument('--topk_match', default=20)
    parser.add_argument('--plot_flag', type=bool, default=True)
    parser.add_argument('--plot_path', type=str, default='plots/')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    main(args)
    print('Done!')