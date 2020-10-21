import torch
import torchvision

import detectron2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.data.datasets import register_pascal_voc
from detectron2.utils import logger

classNames=['Presser Foot', 'Zipper Foot','Sliding Button Foot', 'Blind Hem Foot', 'Bobon', 'Seem Ripper', 'Seem Ripper Lid', 'Screw Driver']

register_pascal_voc("Janome 2212", "../data",'train', 2020, class_names=classNames)