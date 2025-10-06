#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn
from yolox.data import COCODataset

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (608, 544)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (608, 544)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False

        # Define yourself dataset path
        self.data_dir = "/teamspace/studios/this_studio/YOLOX/datasets/unified"
        self.train_ann = "instances_train2017_filtered.json"
        self.val_ann = "instances_val2017_filtered.json"
        self.test_ann = "instances_test2017_filtered.json"


        self.max_epoch = 300          # total number of training epochs
        self.warmup_epochs = 5        # warmup phase (lower LR to stabilize early training)
        self.no_aug_epochs = 15

        self.mosaic_prob = 1.0
        self.mixup_prob = 0.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        self.basic_lr_per_img = 0.01 / 64.0  # base LR per image (scaled by batch size)
        self.scheduler = "yoloxwarmcos"      # LR scheduler

        self.batch_size = 64
        self.eval_interval = 10       # run evaluation every N epochs
        self.num_workers = 4          # dataloader workers

        self.num_classes = 1

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, depthwise=True)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
