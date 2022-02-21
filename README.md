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

### -./result/\*.h5
Well-trained weights of CH-CsiNet. The training dataset is provided by **"C. Wen, W. Shih, and S. Jin, “Deep learning for massive MIMO CSI feedback,” IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748–751, 2018"**

### -./log/\*.out
Log of test/inference process of CH-CsiNet.

**Note that in the submitted paper, we generate the training dataset by ourselves. However, to make this work convincing and facilitate researchers, we use Wen's public dataset in this open source project.**
