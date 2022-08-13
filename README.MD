# **Structure-Aware Motion Transfer with Deformable Anchor Model**
Codes for CVPR 2022 paper Structure-Aware Motion Transfer with Deformable Anchor Model.

## **Environments**
The model are trained with 4 Tesla V100 cards, pytorch with vesion 1.6 and 1.8 are tested fine.

    pip install -r requirements.txt

## **Datasets**
**TaiChiHD**,**Voxceleb1**,**FashionVideo**,**MGIF**, all following [FOMM](https://github.com/AliaksandrSiarohin/first-order-model).

## **Training**
We train the hdam model in two stages. Firstly we train dam, and detect the abnormal keypoints, the indexex of detected abnormal keypoints are written to the hdam config via the "ignore_kp_list" parameter. We then train hdam model with initialization of dam.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run.py --config config/dataset-dam.yaml
    CUDA_VISIBLE_DEVICES=0 python equivariance_detection.py --config config/dataset-dam.yaml --config config/dataset-hdam.yaml --checkpoint path/to/dam/model.pth
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run.py --config config/dataset-hdam.yaml --checkpoint path/to/dam/model.pth
 
## **Evaluation**
Evaluate video reconstruction with following command, for more metrics, we recommend to see [FOMM-Pose-Evaluation](https://github.com/AliaksandrSiarohin/pose-evaluation).

    CUDA_VISIBLE_DEVICES=0 python run.py --mode reconstruction --config path/to/config --checkpoint path/to/model.pth  
## **Demo**
To make a demo animation, specify the driving video and source image, the result video will be saved to result.mp4.

    python demo.py --mode demo --config path/to/config --checkpoint path/to/model.pth --driving_video path/to/video.mp4 --source_image path/to/image.png --result_video path/to/result.mp4 --adapt_scale 

## **Pretrained models**
Coming soon

## **Citation**
    @inproceedings{tao2022structure,
    title={Structure-Aware Motion Transfer with Deformable Anchor Model},
    author={Tao, Jiale and Wang, Biao and Xu, Borun and Ge, Tiezheng and Jiang, Yuning and Li, Wen and Duan, Lixin},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={3637--3646},
    year={2022}
    }

## **Acknowledgements**
The implemetation is heavily borrowed from [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), we thank the author for the great efforts in this area.
