This is a [pytorch](http://pytorch.org/) implementation of **SCT** (Semantic Correlation Transfer for Heterogeneous Domain Adaptation).

#### Prerequisites

```
- Python 3.6
- Pytorch 1.3.1
- numpy
- scipy
- matplotlib
- scikit_learn
- CUDA >= 8.0
```

#### Training Train and Evaluate
Using task  amazon_surf to webcam_decaf as example:
```
python main.py --nepoch 5000 --d_common 256  --combine_pred Cosine_threshold  --cuda 3 --source amazon_surf --target webcam_decaf --lr_first 0.05 --lr 0.006 --mean_loss 1 --shift_loss 0.05 --cos_loss 0.1 --nepoch_first 1500
```

#### Contact

If you have any problem about our code, feel free to contact

- yingzhao@bit.edu.cn
- shuangli@bit.edu.cn
- zhangrui20@bit.edu.cn
- binhuixie@bit.edu.cn

or describe your problem in Issues.