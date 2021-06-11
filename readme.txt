# Certified Radius-Guided Attacks and Efficient
Robustness Training against Deep Neural Networks, IEEE Security & Privacy 2021
\
Wenjie Qu*, Qiming Wu*, Pan Zhou and Binghui Wang


## how to train :
* python train.py --config=configvoc_fcn.json

## how to test(attack):
* python3 test.py --binname=fcn_voc.pth --config=configvoc_fcn.json


## how to test(defense):
* python3 testdefense.py --binname=fcn_voc_def.pth --config=configvoc_fcn.json
