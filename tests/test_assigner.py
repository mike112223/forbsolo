import os
import sys

sys.path.insert(0, os.path.abspath('../forbsolo'))

from forbsolo.utils import Config
from forbsolo.data import build_dataset, build_transform, build_dataloader
from forbsolo.data.dataloader import default_collate
from forbsolo.model import build_model


def main():
    cfg_fp = os.path.join(os.path.abspath('configs'), 'test.py')
    cfg = Config.fromfile(cfg_fp)

    val_tf = build_transform(cfg['data']['val']['transforms'])
    val_dataset = build_dataset(cfg['data']['val']['dataset'], dict(transforms=val_tf))

    val_loader = build_dataloader(cfg['data']['val']['loader'], dict(dataset=val_dataset, collate_fn=default_collate))

    model = build_model(cfg['model']).cuda()

    for i, batch in enumerate(val_loader):
        outputs = model.forward_dummy(batch['img'].cuda())
        break

    img_metas = batch['img_meta']
    gt_bboxes_list = batch['gt_bboxes']
    gt_masks_list = batch['gt_masks']
    gt_labels_list = batch['gt_labels']

    gt_bboxes = gt_bboxes_list[0]
    gt_masks = gt_masks_list[0]
    gt_labels = gt_labels_list[0]
    img_meta = img_metas[0]

if __name__ == '__main__':
    main()
