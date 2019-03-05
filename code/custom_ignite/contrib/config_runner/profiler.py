
import argparse



if __name__ == "__main__":
    pass
    # Run
    # PYTHONPATH=$PYTHONPATH:$PWD/deeplab POLYAXON_NO_OP=1 python3 -m torch.utils.bottleneck custom_ignite/contrib/config_runner/runner.py scripts/training.py deeplab/configs/train/profile_baseline_r18_softmax.py
    # python3 -m cProfile -o /home/storage_ntfs_1tb/output-pascal_voc2012/run_cm_check_shapes_no_ymax_matmul.prof code/custom_ignite/contrib/config_runner/__main__.py code/scripts/training.py code/deeplab/configs/train/baseline_r18_softmax.py
    # snakeviz -H 0.0.0.0 -p 6006 -s run_cm_check_shapes_no_ymax_matmul.prof