

# MTGA: Multi-view Temporal Granularity aligned Aggregation for Event-based Lip-reading

[![Arxiv](https://img.shields.io/badge/Arxiv-2404.11979-red)](https://arxiv.org/pdf/2404.11979)

🔥🔥 2024.12. Our Paper is accepted by AAAI'25!

## Main Requirements 

+ Python 3.8.10
+ PyTorch 2.1.1
+ For other environment requirements, please refer to the environment.yml file.

## Preparation

### 1. dataset preparation

This experiment primarily utilizes the DVS-Lip dataset, which was introduced in [CVPR 2022](https://sites.google.com/view/event-based-lipreading),Please click on the link below to download the dataset, and place the downloaded dataset files in the root directory of your project.

```
DVS-Lip/
|--train/
    |--accused/
|--test/

event-lip/
```

### 2. data preprocessing
In our model, we apply multi-view processing to handle event data. The code already includes the preprocessing steps to convert the raw data into frames, as described in CVPR 2022. Additionally, we transform the event data into voxels. See:

```
event-lip/
|--voxel_utils/
    |--p2v.py
		|--v2g.py
```

These two files provide methods to convert the raw DVS-Lip data into voxel and graph representations. Please note that you will need to set your own data path.

Of course, if you want to quickly start with our code, we also provide pre-converted VoxelList files that you can download using [this link](https://pan.baidu.com/s/1Enq-k92EIzm-NUxfdYPSfg?pwd=ll4c )

### 3. other dependencies installation

We recommend creating a new environment and installing the specified versions of Python and PyTorch before proceeding with the following steps.

please execute the following command:

```
cd event-lip
pip install -r requirements.txt
```

## Training

Once all the data is prepared, a possible directory structure could be as follows:

```
DVS-Lip/
DVS-Lip-VoxelgraphList/
frame_nums.json
event-lip/
```

You can adjust the parameter settings in the fun function in the main.py file according to our model requirements. To run the model, please execute the following command:

```
cd event-lip
python main.py
```