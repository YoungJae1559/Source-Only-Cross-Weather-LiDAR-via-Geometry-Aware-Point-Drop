<h2 align="center">GA-PointDrop</h2>

<p align="center">
  <strong>YoungJae Cheong</strong> ·
  <strong>Jhonhyun An</strong>
  <br>
  <strong>ICRA 2026</strong><br>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2511.01250">
    <strong><code>📄 Paper</code></strong>
  </a>
  <a href="https://github.com/YoungJae1559/GA-PointDrop">
    <strong><code>💻 Source Code</code></strong>
  </a>
</p>

---

## 🔥 News
- [2026/02/01]: 🎉 Our paper has been accepted to the IEEE International Conference on Robotics and Automation (ICRA).
- **Code and pretrained models will be released soon.** (This repository will be updated.)

---

## Abstract
LiDAR semantic segmentation degrades in adverse weather because refraction, scattering, and point dropouts corrupt geometry. Prior work, including weather simulation, mixing-based augmentation, domain randomization, and uncertainty or boundary regularization, improves robustness but still overlooks structural vulnerabilities near boundaries, corners, and sparse regions. We present a Light Geometry-aware adapter. The module aligns azimuth and applies horizontal circular padding to preserve neighbor continuity across the 0◦-360◦ wrap-around boundary. A local-window K-Nearest Neighbors gathers nearby
points and computes simple local statistics, which are compressed into compact geometry-aware cues. During training, these cues drive region-aware regularization that stabilizes predictions in structurally fragile areas. The adapter is plug-and-play, complements augmentation, and can be enabled only
during training with negligible inference cost. We adopt a source-only cross-weather setup where models train on SemanticKITTI and are evaluated on SemanticSTF without target labels or fine-tuning. The adapter improves mIoU by +3.4% over the data-centric augmentation baseline and by +0.3% over the class-centric regularization baseline. These results indicate that geometry-driven regularization is a keydirection for all-weather LiDAR segmentation.

---

## Methods
![method](figs/Figure3.png "model arch")

---

## Installation
```Shell
conda create -n lidar_weather python=3.8 -y && conda activate lidar_weather
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -y
pip install -U openmim && mim install mmengine && mim install 'mmcv>=2.0.0rc4, <2.1.0' && mim install 'mmdet>=3.0.0, <3.2.0'

git clone https://github.com/YoungJae1559/GA-PointDrop.git
cd GA-PointDrop && pip install -v -e .

pip install cumm-cu113 && pip install spconv-cu113
sudo apt-get install libsparsehash-dev
export PATH=/usr/local/cuda/bin:$PATH && pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
pip install nuscenes-devkit
pip install wandb
```

---

## Data Preparation
Please refer to [DATA_PREPARE.md](docs/DATA_PREPARE.md) for the details to prepare the 1SemanticKITTI, 2SynLiDAR, 3SemanticSTF, and 4SemanticKITTI-C datasets.

---

## Getting Started

### SemanticKITTI to SemanticSTF
- [ ] Training & evaluation code for **SemanticKITTI to SemanticSTF**

**Training**
```bash
./tools/dist_train.sh configs/lidarweather_minkunet/sj+lpd+minkunet_semantickitti_GMX.py 2
```

**Evaluation**
```bash
python tools/test.py configs/lidarweather_minkunet/sj+lpd+minkunet_semanticstf_GMX.py sj+lpd+minkunet_semantickitti_GMX/best_miou_epoch_15.pth --task lidar_seg --show-dir /home/vip/harry/LiDARWeather/LiDARweather+GMX --show
```

### SynLiDAR to SemanticSTF
- [ ] Training & evaluation code for **SynLiDAR to SemanticSTF**

**Training**
```bash
./tools/dist_train.sh configs/lidarweather_minkunet/sj+lpd+minkunet_semantickitti_GMX.py 2
```

**Evaluation**
```bash
python tools/test.py configs/lidarweather_minkunet/sj+lpd+minkunet_semanticstf_GMX.py sj+lpd+minkunet_synlidar_GMX/best_miou_epoch_15.pth --task lidar_seg --show-dir /home/vip/harry/LiDARWeather/LiDARweather+GMX --show
```

---

## Contact
- **YoungJae Cheong** — `bluebull777@gachon.ac.kr`
- **Jhonhyun An** — `jhonghyun@gachon.ac.kr`  

> If you have questions, please open an issue or contact us via email.

---

## Related Works
We are deeply grateful for the following outstanding opensource work; without them, our work would not have been possible.
- [LiDARWeather](https://github.com/engineerJPark/LiDARWeather/tree/main)
- [No thing, Nothing](https://github.com/engineerJPark/NTN)
