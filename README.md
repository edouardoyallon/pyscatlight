# pyscatlight_dirt

Installing pyscatlight:

```
pip install -r requirements.txt
python setup.py install
```

A typical call for classification looks like:

```
python main_scatMultiGPU.py PATHIMAGENET --arch scat50
```
It is also possible to employ main_scatDistributed.py which requires NCCL.

For detection, a pretrained model can be found here: https://s3.amazonaws.com/modelzoo-networks/scatresnet50.tar.gz

A typical call for detection looks like:

```
 python trainval_net_scat.py  --dataset pascal_voc --net scatnet  --cuda --save_dir FOLDER --nw 4 --lr 1e-2 --lr_decay_step 6 --bs  3 --epochs 10 --s 3 --mGPUs
```

Our detection pipeline relies on https://github.com/jwyang/faster-rcnn.pytorch . It was successfully re-tested for pascal
and coco using pytorch 0.4.0, with python 3.5 . For
the sake of reproducibility, the files that were modified or created at time of submission were:

detection/lib/roi_data_layer/roibatchLoader.py

detection/trainval_net_scat.py

detection/test_net_scat.py

detection/lib/model/utils/blob.py - has to be modified for COCO

detection/cfgs/scatnet.yml

detection/cfgs/scatnet_ls.yml

detection/lib/model/faster_rcnn/scatnet.py