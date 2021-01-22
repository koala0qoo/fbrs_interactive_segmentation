import argparse
import json
import torch
import numpy as np
import cv2
from torchvision import transforms
from isegm.utils import exp
from isegm.inference import utils
from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.inference.utils import structural_integrity_strategy


def main():
    args, cfg = parse_args()

    # get model
    torch.backends.cudnn.deterministic = True
    checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    model = utils.load_is_model(checkpoint_path, args.device, cpu_dist_maps=True, norm_radius=args.norm_radius)
    predictor = get_predictor(model.to(args.device), device=args.device,
                              brs_mode=args.mode)
    clicker = clicker_.Clicker()
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    # get image
    image = cv2.cvtColor(cv2.imread(args.imgpath), cv2.COLOR_BGR2RGB)
    image_nd = input_transform(image).to(args.device)
    if image_nd is not None:
        predictor.set_input_image(image_nd)

    probs_history = []
    clicks = json.load(open('clicks.json','r'))['clicks']
    for c in clicks:
        click = clicker_.Click(is_positive=c['is_positive'], coords=(c['y'], c['x']))
        clicker.add_click(click)
        pred = predictor.get_prediction(clicker)
        torch.cuda.empty_cache()

        if probs_history:
            probs_history.append((probs_history[-1][0], pred))
        else:
            probs_history.append((np.zeros_like(pred), pred))

    if probs_history:
        current_prob_total, current_prob_additive = probs_history[-1]
        final_pred = np.maximum(current_prob_total, current_prob_additive)
        final_mask = (final_pred > 0.5).astype(np.uint8)
    else:
        final_mask = np.ones_like(pred).astype(np.uint8)

    if args.sis:
        final_mask = structural_integrity_strategy(final_mask, clicker)


    cv2.imwrite(args.outpath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR) * np.expand_dims(final_mask, axis=2))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--img', type=str, default='./test.jpg',
                        help='The path to the image. ')

    parser.add_argument('--gpu', type=int, default=0,
                        help='Id of GPU to use.')

    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use only CPU for inference.')

    parser.add_argument('--limit-longest-size', type=int, default=800,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')

    parser.add_argument('--norm-radius', type=int, default=260)

    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')

    parser.add_argument('--mode', default='f-BRS-B', choices=['NoBRS', 'RGB-BRS', 'DistMap-BRS',
                                         'f-BRS-A', 'f-BRS-B', 'f-BRS-C'],
                        help='')

    parser.add_argument('--sis', action='store_true', default=False,
                        help='Use sis.')

    args = parser.parse_args()
    if args.cpu:
        args.device =torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')
    cfg = exp.load_config_file(args.cfg, return_edict=True)

    return args, cfg


if __name__ == '__main__':
    main()
