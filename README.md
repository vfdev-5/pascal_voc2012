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

### [Pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception.git)

- Initialize submodule:
```bash
git submodule update --init --recursive
```

- Run the training 
```bash
polyaxon run -u -f plx_configs/pytorch-deeplab-xception/xp_train_voc.yaml --name="pytorch-deeplab-xception-train-voc" --tags=pytorch-deeplab-xception
```

