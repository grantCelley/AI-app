import torch
import torchvision
from detectron2.data.datasets import register_pascal_voc

classNames=['Presser Foot', 'Zipper Foot','Sliding Button Foot', 'Blind Hem Foot', 'Bobon', 'Seem Ripper', 'Seem Ripper Lid', 'Screw Driver']

register_pascal_voc("Janome 2212", "../data",'train', 2020, class_names=classNames)
