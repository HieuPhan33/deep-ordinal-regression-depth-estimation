# DORN implemented in Pytorch 1.0.0


### Introduction
This is a PyTorch 1.0.0 for DORN with self-attention module
### Configuration:
You can set the total_ordinal_numbers, image_size for each type of dataset in ``config.py``

### Prepare dataset:
Place data in `data_dir`: `root_dir/data/dataset_name` <br>
Organize train and val in `data_dir/train` and `data_dir/val`
dataset_name must be 'nyu','make3d','kitti','uow_dataset_full', please name your folders appropriately <br>
Change variable `data_dir` on line 53 in `main.py` as `root_dir` <br>
Link for make3d and uow: www.kaggle.com/dataset/b0fc6abcf37d10d62182779cc37187b52016aaa9b81fe824bd76c2ab4661f279 <br>

### Training:
python main.py --batch-size b -- dataset dataset_name --gpu 0 --epochs 20
Set b=4 to avoid run-out-of-memory
Start with make3d and uow_dataset_full
dataset_name must be either 'nyu','make3d','kitti','uow_dataset_full'
