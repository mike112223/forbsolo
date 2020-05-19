import os
import sys

sys.path.insert(0, os.path.abspath('../forbsolo'))

from forbsolo.utils import Config
from forbsolo.data import build_dataset, build_transform, build_dataloader
from forbsolo.data.dataloader import default_collate
from forbsolo.model import build_model


# def main():
cfg_fp = os.path.join(os.path.abspath('config'), 'solo_test.py')
cfg = Config.fromfile(cfg_fp)

val_tf = build_transform(cfg['data']['val']['transforms'])
val_dataset = build_dataset(cfg['data']['val']['dataset'], dict(transforms=val_tf))

val_loader = build_dataloader(cfg['data']['val']['loader'], dict(dataset=val_dataset, collate_fn=default_collate))

model = build_model(cfg['model']).cuda()

for i, batch in enumerate(val_loader):
    img = batch['img']
    img_metas = batch['img_meta']
    gt_bboxes_list = batch['gt_bboxes']
    gt_masks_list = batch['gt_masks']
    gt_labels_list = batch['gt_labels']
    gt_bboxes_ignore_list = batch['gt_bboxes_ignore']

    pred_results, target_results = model.forward_train(
        img.float().cuda(),
        img_metas,
        gt_bboxes_list,
        gt_masks_list,
        gt_labels_list,
        gt_bboxes_ignore_list)

    break

if __name__ == '__main__':
    main()
