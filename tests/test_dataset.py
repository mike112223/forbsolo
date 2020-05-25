import os
import sys
sys.path.insert(0, os.path.abspath('../forbsolo'))

import cv2
import numpy as np

from forbsolo.utils import Config
from forbsolo.data import build_dataset, build_transform, build_dataloader
from forbsolo.data.dataloader import collate


cat_color = np.random.randint(0, 255, (80, 3))


def main():
    cfg_fp = os.path.join(os.path.abspath('config'), 'solo_r50_fpn_single_gpu.py')
    cfg = Config.fromfile(cfg_fp)

    val_tf = build_transform(cfg['data']['val']['transforms'])
    val_dataset = build_dataset(cfg['data']['val']['dataset'], dict(transforms=val_tf))

    val_loader = build_dataloader(cfg['data']['val']['loader'], dict(dataset=val_dataset, collate_fn=collate))

    for i, batch in enumerate(val_loader):
        print(batch['img'].numpy().shape)
        print('---------------------')
        for j in range(len(batch['img'].numpy())):
            print(j)
            img = batch['img'].numpy()[j].transpose(1, 2, 0)

            # print(img.dtype)
            gt_bboxes = batch['gt_bboxes'][j].numpy()
            gt_labels = batch['gt_labels'][j].numpy()
            gt_masks = batch['gt_masks'][j].numpy()
            img_meta = batch['img_meta'][j]

            dmasks = np.zeros(img.shape, np.uint8)

            for j, gt_bbox in enumerate(gt_bboxes):

                gt_mask = gt_masks[j]
                dmasks[gt_mask == 1] = cat_color[gt_labels[j]]

                img = cv2.rectangle(
                    img,
                    (gt_bbox[0], gt_bbox[1]),
                    (gt_bbox[2], gt_bbox[3]),
                    (0, 255, 0),
                    2
                )

                img = cv2.putText(
                    img,
                    '%d' % gt_labels[j],
                    (gt_bbox[0], gt_bbox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 0, 0),
                    2
                )

            img = cv2.addWeighted(img, 1., dmasks, 0.4, 0)

            save_path = os.path.join('tests/test_imgs/', img_meta['filename'].split('/')[-1])

            cv2.imwrite(save_path, img)

            with open(save_path[:-3] + 'txt', 'w') as f:
                f.write(str(img_meta))

        if i == 10:
            break

if __name__ == '__main__':
    main()
