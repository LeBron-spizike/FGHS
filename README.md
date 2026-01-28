#Few-Shot Molecular Property Prediction via Fine-Grained and Hierarchical Structure Representation Learning

This repository is the implementation of FGHS (Few-Shot Molecular Property Prediction via Fine-Grained and Hierarchical Structure Representation Learning).

## Framework
<img src="framework.png" alt="framework" style="zoom: 100%;" />
 


## Environment

To run the code successfully, the following dependencies need to be installed:

```
python           3.7
torch            1.13.1
torch_geometric  2.3.1
torch_scatter    2.1.1
rdkit            2023.3.2
```

## Implementation

### Datasets

For data used in the experiments, please save the contents in the `data` directory.


### Usage

Under the 10-shot setting:

```sh
python run.py --dataset {dataset} --n_support 10 --train_auxi_task_num {num} --test_auxi_task_num {num}
```

Under the 1-shot setting:

```sh
python run.py --dataset {dataset} --n_support 1 --train_auxi_task_num {num} --test_auxi_task_num {num}
```
