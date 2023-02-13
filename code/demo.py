from argparse import ArgumentParser
from pathlib import Path

import os
import cv2
import numpy as np
from collections import OrderedDict

import torch
from torchvision.transforms.functional import to_tensor

from tqdm import tqdm

from models.model import GLPDepth
from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


def main(src, dst, device='cuda', model_path=Path('./ckpt/best_model_nyu.ckpt')):
    model = GLPDepth(max_depth=10, is_train=False).to(device)
    model_weight = torch.load(model_path, map_location=device)
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    pbar = tqdm(total=nframes, desc='depth estimation')
    writer = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape
        new_h, new_w = h // 32 * 32, w // 32 * 32
        frame = cv2.resize(frame, (new_w, new_h))
        batch = to_tensor(frame).to(device).unsqueeze(0)
        with torch.no_grad():
            pred = model(batch)['pred_d']

        depth = pred.squeeze().cpu().numpy()
        depth = (depth / depth.max()) * 255
        depth = depth.astype(np.uint8)
        depth = cv2.resize(depth, (w, h))
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_RAINBOW)

        if writer is None:
            writer = cv2.VideoWriter(
                str(dst),
                cv2.VideoWriter_fourcc(*'avc1'),
                fps,
                (w, h)
            )
        writer.write(depth)
        pbar.update(1)
    writer.release()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('src', type=Path)
    parser.add_argument('dst', type=Path)
    parser.add_argument('--device', type=str, default='cuda')
    clargs = parser.parse_args()

    main(**vars(clargs))
