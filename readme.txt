how to train :
python train.py --config=configvoc_fcn.json

how to test(attack):
python3 test.py --binname=fcn_voc.pth --config=configvoc_fcn.json


how to test(defense):
python3 testdefense.py --binname=fcn_voc_def.pth --config=configvoc_fcn.json
