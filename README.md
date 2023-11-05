# COMP8539 Group Project - Zhen Wang, Tony Chen, Jipei Chen

We modified the code based on the official implementation for [3D Human Pose Estimation with Spatial and Temporal Transformers](https://arxiv.org/pdf/2103.10455.pdf).
- We recorded five improvement version in the release notes
  - original version as baseline
  - self-attention & multi-head
  - dct
  - wavelet
  - upsampling
  - gradient
- We use two visualization tool kit from videoPose3D and PoseFormerV2 and adapt to our use case
- We did not update the large .bin file for the final model result. We uploaded all the log files that records the evaluation results of those bin files.

- Inference Video Demonstration 
    - [sample version1](https://github.com/WangZhen-Ryan/PoseFormerV1-COMP8539/blob/main/visualisation/01/data/output.mp4)
    - [sample version2](https://drive.google.com/drive/folders/1R0PXb9Y1ninF9YuRj11Y8vhSPi6su7IZ?usp=sharing) OR [full sample output here](https://drive.google.com/file/d/1eAhh8z3zT72M4_P_pthLMQL8vXvli2w9/view?usp=sharing)

### Environment

The code is developed and tested under the following environment

* Python 3.8.2
* PyTorch 1.7.1
* CUDA 11.0

You can create the environment:
```bash
conda env create -f poseformer.yml
```

### Dataset

 Please refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset  (./data directory). 

### Training new models

* default lr is 0.00002 and lrd is 0.98
* To train a model from scratch (CPN detected 2D pose as input), run:

```bash
python run_poseformer.py -k cpn_ft_h36m_dbb -f 9
```

`-f` controls how many frames are used as input. 27 frames achieves 47.0 mm, 81 frames achieves achieves 44.3 mm. 

* To train a model from scratch (Ground truth 2D pose as input), run:

```bash
python run_poseformer.py -k gt -f 9
```

### Evaluating trained models

Evaluate the 9-frame model (CPN detected 2D pose as input), put it into the `./checkpoint` directory and run:

```bash
python run_poseformer.py -k cpn_ft_h36m_dbb -f 9 -c checkpoint --evaluate NAME_OF_MODEL.bin
```

Evaluate the 9-frame model (Ground truth 2D pose as input), put it into the `./checkpoint` directory and run:

```bash
python run_poseformer.py -k gt -f 9 -c checkpoint --evaluate NAME_OF_MODEL.bin
```


