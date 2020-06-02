import torch
import os.path as osp
import numpy as np
from collections.abc import Iterable

from forbsolo.utils.checkpoint import load_checkpoint, save_checkpoint

from .registry import RUNNERS


@RUNNERS.register_module
class Runner(object):
    """ Runner

    """
    def __init__(self,
                 loader,
                 model,
                 optim,
                 lr_scheduler,
                 max_epochs,
                 workdir,
                 start_epoch=0,
                 trainval_ratio=1,
                 snapshot_interval=1,
                 gpu=True,
                 test_cfg=None,
                 test_mode=False,
                 print_freq=50):
        self.loader = loader
        self.model = model
        self.metric = None
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.start_epoch = start_epoch
        self.max_epochs = max_epochs
        self.workdir = workdir
        self.trainval_ratio = trainval_ratio
        self.snapshot_interval = snapshot_interval
        self.gpu = gpu
        self.test_cfg = test_cfg
        self.test_mode = test_mode
        self.print_freq = print_freq

    def __call__(self):
        if self.test_mode:
            self.test_epoch()
        else:
            assert self.trainval_ratio > 0
            for epoch in range(self.start_epoch, self.max_epochs):
                self.train_epoch()
                self.save_checkpoint(self.workdir)
                if self.trainval_ratio > 0 \
                        and (epoch + 1) % self.trainval_ratio == 0 \
                        and self.loader.get('val'):
                    self.val_epoch()

    def train_epoch(self):
        print('Epoch %d, Start training' % self.epoch)
        iter_based = hasattr(self.lr_scheduler, '_iter_based')
        for batch in self.loader['train']:

            self.train_batch(batch)
            if iter_based:
                self.lr_scheduler.step()
        if not iter_based:
            self.lr_scheduler.step()

    def train_batch(self, batch):
        self.model.train()

        self.optim.zero_grad()

        seg_losses, cls_losses = self.model(**batch)
        losses = sum(seg_losses + cls_losses)

        losses.backward()

        self.optim.step()

        if self.iter != 0 and self.iter % self.print_freq == 0:
            print(
                'Train, Epoch[%d][%d/%d], lr %s, cls loss %.4f, seg loss %.4f, total loss: %.4f' %
                (self.epoch, self.iter, len(self.loader['train']), self.lr, sum(cls_losses).item(),
                 sum(seg_losses).item(), losses.item()))

    def val_epoch(self):
        print('Epoch %d, Start validating' % self.epoch)
        for batch in self.loader['val']:
            self.val_batch(batch)

    def val_batch(self, batch):
        self.model.eval()
        with torch.no_grad():

            seg_losses, cls_losses = self.model(**batch)
            losses = sum(seg_losses + cls_losses)

            if self.iter != 0 and self.iter % 10 == 0:
                print(
                    'Val, Epoch[%d][%d/%d], lr %s, cls loss %.4f, seg loss %.4f, total loss: %.4f' %
                    (self.epoch, self.iter, len(self.loader['val']), self.lr, sum(cls_losses).item(),
                     sum(seg_losses).item(), losses.item()))

    def test_epoch(self):
        print('Epoch %d, Start testing' % self.epoch)

        results = []
        gt_bboxes = []
        gt_labels = []

        for batch in self.loader['test']:
            results.append(self.test_batch(batch))
            gt_bboxes.append(batch['gt_bboxes'])
            gt_labels.append(batch['gt_labels'])

        # eval_map(results,
        #          gt_bboxes,
        #          gt_labels,
        #          gt_ignore=None,
        #          scale_ranges=None,
        #          iou_thr=0.5,
        #          dataset=self.loader.dataset,
        #          print_summary=True)

    def test_batch(self, batch):
        self.model.eval()
        with torch.no_grad():

            img_metas = batch['img_meta']
            # print([_['filename'] for _ in img_metas])
            imgs = batch['img']

            if self.gpu:
                imgs = imgs.cuda()

            bbox_results = self.model(
                imgs,
                img_metas,
                self.test_mode,
            )

        return bbox_results

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):
        if self.epoch % self.snapshot_interval == 0 or self.epoch == self.max_epochs:
            if meta is None:
                meta = dict(epoch=self.epoch, iter=self.iter, lr=self.lr)
            else:
                meta.update(epoch=self.epoch, iter=self.iter, lr=self.lr)

            filename = filename_tmpl.format(self.epoch)
            filepath = osp.join(out_dir, filename)
            linkpath = osp.join(out_dir, 'latest.pth')
            optimizer = self.optim if save_optimizer else None
            print('Save checkpoint %s', filename)
            save_checkpoint(self.model,
                            filepath,
                            optimizer=optimizer,
                            meta=meta)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        print('Resume from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict)

    @property
    def epoch(self):
        """int: Current epoch."""
        return self.lr_scheduler.last_epoch

    @epoch.setter
    def epoch(self, val):
        """int: Current epoch."""
        self.lr_scheduler.last_epoch = val

    @property
    def lr(self):
        lr = [x['lr'] for x in self.optim.param_groups]
        return np.array(lr)

    @lr.setter
    def lr(self, val):
        for idx, param in enumerate(self.optim.param_groups):
            if isinstance(val, Iterable):
                param['lr'] = val[idx]
            else:
                param['lr'] = val

    @property
    def iter(self):
        """int: Current iteration."""
        return self.lr_scheduler.last_iter

    @iter.setter
    def iter(self, val):
        """int: Current epoch."""
        self.lr_scheduler.last_iter = val
