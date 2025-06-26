# Mixed-Integer Linear Optimization via Learning-Based Two-Layer Large Neighborhood Search

This repository is an implementation of the LION19 paper: Mixed-Integer Linear Optimization via Learning-Based Two-Layer Large Neighborhood Search.

## Software dependencies

```markdown
python=3.11
pytorch=2.1.0
scip=8.0.4
gurobi=10.0.3
torch-geometric=2.4.0
pytorch-metric-learning=2.5.0
```

## Running experiments

Use the following command to generate training/testing instances:

```markdown
python instance_generate.py --usage 'test' --instance 'SC' --number 10
python instance_generate.py --usage 'train' --instance 'SC' --number 1000
```

For training data collection:

```markdown
python data_collection.py
```

For training model:

```markdown
python train.py
```

You can directly utilize the trained model in this repo, and for inference, use the following command:

```markdown
bash run.sh
```

For the comparison:
```markdown
python result.py
```

## Citing our work

If you would like to utilize TLNS in your research, we kindly request that you cite our paper as follows:

```text
@article{liu2024mixed,
  title={Mixed-Integer Linear Optimization via Learning-Based Two-Layer Large Neighborhood Search},
  author={Liu, Wenbo and Wang, Akang and Yang, Wenguo and Shi, Qingjiang},
  journal={arXiv preprint arXiv:2412.08206},
  year={2024}
}
```
