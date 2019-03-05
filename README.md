# NN on Pascal VOC2012

## Project on Polyaxon platform

### Create project
```
polyaxon project create --name=pascal-voc2012 --description="Semantic segmentation on Pascal VOC2012"
```

### Initialize project 
```
polyaxon init pascal-voc2012
```

## Content


## How to run without cluster/Polyaxon


```bash
export DATASET_PATH=/path/to/dataset/ 
export OUTPUT_PATH=/path/to/output
export POLYAXON_NO_OP=1 
export PYTHONPATH=$PYTHONPATH:$PWD/code/:$PWD/code/deeplab
python3 -m custom_ignite.contrib.config_runner code/scripts/training.py code/deeplab/configs/train/baseline_r18_softmax.py
```