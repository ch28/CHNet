# CHNet

**Open source codes for paper "Changeable Rate and Novel Quantization for CSI Feedback Based on Deep Learning"**

Environment Requirement:
python == 3.5.2

tensorflow-gpu == 1.12.0

keras == 2.2.4

### CH-CsiNet_train.py
Training code for CH-CsiNet, a (**ch**)angeable Rate CSI feedback network based on CsiNet.

### CH-CsiNet_test.py
Test/Inference code for CH-CsiNet.

### CsiNet-PQB.py
CsiNet with bump function-based approximate quantization gradient.

### ./result/\*.h5
Well-trained weights of CH-CsiNet. The training dataset is provided by **"C. Wen, W. Shih, and S. Jin, “Deep learning for massive MIMO CSI feedback,” IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748–751, 2018"**

### ./log/\*.out
Log of test/inference process of CH-CsiNet.

### CRNet.py
Tensorflow version of CRNet for paper "Z. Lu, J. Wang and J. Song, ”Multi-resolution CSI Feedback with Deep Learning in Massive MIMO System,” ICC 2020 - 2020 IEEE International Conference on Communications (ICC), 2020, pp. 1-6, doi: 10.1109/ICC40277.2020.9149229."

The authors of CRNet also provide the open source codes of CRNet at https://github.com/Kylin9511/CRNet

**Note that in the submitted paper, we generate the training dataset by ourselves. However, to make this work convincing and facilitate researchers, we use Wen's public dataset in this open source project.**
