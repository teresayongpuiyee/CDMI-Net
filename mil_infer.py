# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""

import os
import torch
from torch.autograd import Variable
from mil_model import Attention
import numpy as np
import util
import accuracy as acc
from tqdm import tqdm
import time
import argparse
import torch.utils.data as data_utils
from mil_dataloader import CDBags
from torchmetrics import Accuracy, Precision, Recall, F1Score, StatScores, JaccardIndex
import pandas as pd


def eval_img(model, args, label, file_name, gt_path, img_gt_pred, accuracy_metric, precision_metric, recall_metric, f1_metric, stats_metric, iou_metric, pixel_accuracy_metric, pixel_precision_metric, pixel_recall_metric, pixel_f1_metric):
    time_start_all = time.time()

    dim = 256

    t1_path = file_name + 'T1.tif'
    t2_path = file_name + 'T2.tif'

    img_name = file_name.split("/")[-1][:-1] + ".tif"
    
    out_path_flse = os.path.join(args.save_dir, 'flse', img_name)
    #out_path_score = os.path.join(args.save_dir, 'score',img_name)
    #out_path_bm = os.path.join(args.save_dir, 'bm', img_name)
    #out_path_scene = os.path.join(args.save_dir, 'scene', img_name)
    out_dataset_flse = util.create_tiff(out_path_flse, t1_path)
    #out_dataset_score = util.create_tiff(out_path_score, t1_path)
    #out_dataset_bm = util.create_tiff(out_path_bm, t1_path)
    #out_dataset_scene = util.create_tiff(out_path_scene, t1_path)

    t1 = util.read_tiff(t1_path)
    t2 = util.read_tiff(t2_path)
    w = t1.RasterXSize
    h = t1.RasterYSize
    h_batch = int(h / dim)
    w_batch = int(w / dim)
    all_count = h_batch * w_batch
    hist = np.zeros((2, 2))
    if len(gt_path) > 1:
        gt = util.read_tiff(gt_path)

    for index in tqdm(range(all_count)):
        i = int(index / w_batch)  # row
        j = index % w_batch  # col
        x = j * dim
        y = i * dim
        t1_b = util.read_block(t1, x, y, dim)
        t2_b = util.read_block(t2, x, y, dim)
        t2_b = util.hist_match(t2_b, t1_b)

        if len(gt_path) > 1:
            gt_b = util.read_block(gt, x, y, dim)
            gt_b[gt_b < 255] = 0
            #gt_b[gt_b == 255] = 0
            #gt_b[gt_b > 0] = 255

        data1 = t1_b.transpose((2, 0, 1))
        data2 = t2_b.transpose((2, 0, 1))

        data1 = data1[np.newaxis, ...]
        data2 = data2[np.newaxis, ...]

        data_v_1 = Variable(torch.from_numpy(data1))
        data_v_2 = Variable(torch.from_numpy(data2))

        if not args.no_gpu:
            data_v_1 = data_v_1.cuda()
            data_v_2 = data_v_2.cuda()

        model.train()
        pred_prob, pred_label, attention_weights = model.eval_img(
            data_v_1, data_v_2)
        
        preds = torch.from_numpy(pred_label)
        cls_label = label[0]

        # Update each metric with the current batch predictions and labels
        accuracy_metric.update(preds, cls_label)
        precision_metric.update(preds, cls_label)
        recall_metric.update(preds, cls_label)
        f1_metric.update(preds, cls_label)
        stats_metric.update(preds, cls_label)  # Update stats metric

        img_name_png = file_name.split("/")[-1][:-1] + ".png"
        img_gt_pred.append({"image_name": img_name_png, "cls_label": int(cls_label.item()), "pred_label": int(preds.item())})

        model.eval()
        if pred_label[0] > 0.5:
            pred_label = 'P'
            bmm = np.ones((dim, dim)) * 255
            weight = attention_weights.data[0].cpu().detach().numpy()
            weight = weight.reshape((dim, dim))
            cmm = weight * 255.0 / np.max(weight)
            bm = cmm.copy()
            bm[bm < 128] = 0
            bm[bm > 0] = 255
            cva = np.abs(t1_b - t2_b)
            cva = np.power(cva, 2)
            cva = np.sum(cva, axis=2)
            cva = cva / 3.0
            cva = np.sqrt(cva)
            flse = util.FLSE(
                cva,
                bm,
                args.sigma,
                args.gaussian,
                args.delt,
                args.iter)
            flse = np.asarray(flse, dtype=np.uint8)
            flse[flse > 0] = 255
            flse[flse <= 0] = 0

            sub_dir = os.path.join(args.save_dir)
            b1_path = os.path.join(sub_dir, "r{0}_c{1}_b1.tif".format(i, j))
            #util.save_map(b1_path, t1_b)

            b2_path = os.path.join(sub_dir, "r{0}_c{1}_b2.tif".format(i, j))
            #util.save_map(b2_path, t2_b)
            if len(gt_path) > 1:
                bg_path = os.path.join(
                    sub_dir, "r{0}_c{1}_gt.tif".format(i, j))
                #util.save_map(bg_path, gt_b)

            bm_path = os.path.join(sub_dir, "r{0}_c{1}_bm.tif".format(i, j))
            #util.save_map(bm_path, bm)

            score_path = os.path.join(
                sub_dir, "r{0}_c{1}_score.tif".format(i, j))
            #util.save_map(score_path, cmm)

            flse_path = os.path.join(
                sub_dir, "r{0}_c{1}_flse.tif".format(i, j))
            #util.save_map(flse_path, flse)

        else:
            pred_label = 'N'
            cmm = np.zeros((dim, dim))
            bm = np.zeros((dim, dim))
            flse = np.zeros((dim, dim))
            bmm = np.zeros((dim, dim))

        util.write_block(out_dataset_flse, flse, x, y, dim)
        #util.write_block(out_dataset_score, cmm, x, y, dim)
        #util.write_block(out_dataset_bm, bm, x, y, dim)
        #util.write_block(out_dataset_scene, bmm, x, y, dim)
        if len(gt_path) > 1:
            #gt_pixel_count = np.count_nonzero(gt_b)
            if cls_label.item() == 1:
                if pred_label == 'P':
                    hist[0, 0] = hist[0, 0] + 1.0  # TP
                else:
                    hist[1, 0] = hist[1, 0] + 1.0  # FN
            elif cls_label.item() == 0:
                if pred_label == 'P':
                    hist[0, 1] = hist[0, 1] + 1.0  # FP
                else:
                    hist[1, 1] = hist[1, 1] + 1.0  # TN

    del out_dataset_flse
    #del out_dataset_score
    #del out_dataset_bm
    #del out_dataset_scene

    if len(gt_path) > 1:
        time_end_all = time.time()
        print('All time {:.2f}'.format(time_end_all - time_start_all))
        print("CDMI-Net: Scene-based accuracy")
        _ = acc.evaluation_print(hist, scene=True)

        gt_data = util.read_image(gt_path)
        gt_data[gt_data == 255] = 1
        #gt_data[gt_data == 255] = 0
        pred_data = util.read_image(out_path_flse)
        acc_matrix = acc.hist(gt_data, pred_data)
        print("CDMI-Net: Pixel-based accuracy")
        acc.evaluation_print(acc_matrix)

        # Convert pred_data and gt_data to PyTorch tensors and add batch dimension
        pred_tensor = torch.tensor(pred_data.astype(int)).unsqueeze(0)  # Shape: (1, 256, 256)
        gt_tensor = torch.tensor(gt_data.astype(int)).unsqueeze(0)      # Shape: (1, 256, 256)

        # Update IoU metric with current batch predictions and targets
        iou_metric.update(pred_tensor, gt_tensor)
        pixel_accuracy_metric.update(pred_tensor, gt_tensor)
        pixel_precision_metric.update(pred_tensor, gt_tensor)
        pixel_recall_metric.update(pred_tensor, gt_tensor)
        pixel_f1_metric.update(pred_tensor, gt_tensor)

    return img_gt_pred


if __name__ == "__main__":
    '''
    python mil_infer.py --data_dir /home/jovyan/change_detection/data/bottle-cdmi --save_dir bottle/e30_iter20_final --weight

    '''
    args = argparse.ArgumentParser(description='Start inference stage ...')
    args.add_argument('--data_dir', required=True, help='Training set dir.')
    args.add_argument('--seed', type=int, default=1, help='Random seed.')
    args.add_argument('--weight', required=True, help='Check point path.')
    args.add_argument('--save_dir', required=True, help='Output dir.')

    args.add_argument(
        '--sigma',
        type=int,
        default=1,
        help='Parameter [sigma] of FLSE.')
    args.add_argument('--gaussian', type=int, default=9,
                      help='Parameter [gaussian_size] of FLSE.')
    args.add_argument(
        '--delt',
        type=int,
        default=8,
        help='Parameter [sigma] of FLSE.')
    args.add_argument(
        '--iter',
        type=int,
        default=20,
        help='Parameter [iter] of FLSE.')
    args.add_argument('--no-gpu', action='store_true', help='Using CPU.')

    args = args.parse_args()
    args.save_dir = "./outputs/" + args.save_dir + "/prediction"
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    os.makedirs(args.save_dir+'/flse', exist_ok = True)
    #os.makedirs(args.save_dir+'/score', exist_ok = True)
    #os.makedirs(args.save_dir+'/bm', exist_ok = True)
    #os.makedirs(args.save_dir+'/scene', exist_ok = True)

    args_gpu = not args.no_gpu and torch.cuda.is_available()
    
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args_gpu else {}

    test_loader = data_utils.DataLoader(CDBags(data_dir=args.data_dir,
                                               seed=args.seed,
                                               train=False),
                                        batch_size=1, # batch size must be 1
                                        shuffle=False,
                                        **loader_kwargs)

    model = Attention()
    model.load_state_dict(torch.load(args.weight))
    # model.print_size()
    if not args.no_gpu:
        model.cuda()

    img_gt_pred = []
    # Initialize metric objects
    accuracy_metric = Accuracy(task="binary").to('cuda')
    precision_metric = Precision(task="binary", zero_division=1).to('cuda')
    recall_metric = Recall(task="binary", zero_division=1).to('cuda')
    f1_metric = F1Score(task="binary", zero_division=1).to('cuda')
    stats_metric = StatScores(task="binary").to('cuda')  # To get TP, TN, FP, FN

    # Initialize the IoU metric
    iou_metric = JaccardIndex(task="multiclass", num_classes=2, average="none")  # Use average='none' for class-wise IoU
    pixel_accuracy_metric = Accuracy(task="multiclass", num_classes=2)
    pixel_precision_metric = Precision(task="multiclass", num_classes=2, average="none")
    pixel_recall_metric = Recall(task="multiclass", num_classes=2, average="none")
    pixel_f1_metric = F1Score(task="multiclass", num_classes=2, average="none")
    iou_metric.reset()

    for batch_idx, (_, _, label, file_name, gt_path) in enumerate(test_loader):
        img_gt_pred = eval_img(model, args, label, file_name[0], gt_path[0], img_gt_pred, accuracy_metric, precision_metric, recall_metric, f1_metric, stats_metric, iou_metric, pixel_accuracy_metric, pixel_precision_metric, pixel_recall_metric, pixel_f1_metric)
    
    # Compute final metrics after the loop
    accuracy = accuracy_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1 = f1_metric.compute()
    # Get TP, TN, FP, FN from the stats metric
    tp, fp, tn, fn, _ = stats_metric.compute()

    # Compute the final IoU across the entire validation set
    iou_score = iou_metric.compute()
    pixel_accuracy = pixel_accuracy_metric.compute()
    pixel_precision = pixel_precision_metric.compute()
    pixel_recall = pixel_recall_metric.compute()
    pixel_f1 = pixel_f1_metric.compute()

    # Convert to DataFrame
    df = pd.DataFrame(img_gt_pred)

    # Save to CSV
    df.to_csv(args.save_dir + "/output.csv", index=False)
    
    cls_score = {"accuracy": float(accuracy.item()), "precision": float(precision.item()), "recall": float(recall.item()), "f1": float(f1.item()), "tp": int(tp.item()), "fp": int(fp.item()), "tn": int(tn.item()), "fn": int(fn.item())}
    pixel_score = {"iou_score": iou_score, "accuracy": pixel_accuracy, "precision": pixel_precision, "recall": pixel_recall, "f1": pixel_f1}

    # Print or log metrics
    print("cls_score:")
    print(cls_score)

    print("pixel_score:")
    print(pixel_score)

    print('Done!')
