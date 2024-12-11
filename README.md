![FedIIH Logo](FedIIH_logo.png)

# Modeling Inter-Intra Heterogeneity for Graph Federated Learning

Official Code Repository for our paper (AAAI 2025) :

Modeling Inter-Intra Heterogeneity for Graph Federated Learning


## Highlights
- User-Friendly: Simple and intuitive to use.
- SOTA Performance: Outperforms the second-best by 5.79% on all heterophilic datasets.
- Rich Datasets: Covers 11 GFL datasets (6 homophilic, 5 heterophilic).

## Requirement
- Python 3.8.8
- PyTorch 1.12.0+cu113
- PyTorch Geometric 2.3.0
- METIS (only for subgraph generation) https://github.com/james77777778/metis_python

## Subgraph generation
Download from the Google Drive (https://drive.google.com/file/d/1PyqvR6yL43Om42fdsbKHj5WCgREvi3St/view?usp=sharing) and then unzip it.

Place the `datasets` folder in the same path as `README.md`

or

Follow command lines automatically to generate the subgraphs.
```sh
$ cd FedIIH_2/data/generators
$ python disjoint.py
$ python overlapping.py
```

## Parameter description


- `gpus`: specify gpus to use
- `num workers`: specify the number of workers on gpus (_e.g._, if your experiment uses 10 clients for every round then use less than or equal to 10 workers). The actual number of workers will be `num_workers` + 1 (one additional worker for a server).
- `FedIIH_2` means that the number of disentangled latent factors is set to 2 (K=2). Similarly, `FedIIH_10` means that the number of disentangled latent factors is set to 10 (K=10).

 

# Homophilic datasets
Follow command lines to run the experiments.
## Cora
### non-overlapping
```Python
$ cd FedIIH_2
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Cora --mode disjoint --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```
### overlapping
```Python
$ cd FedIIH_2
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Cora --mode overlapping --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```


## CiteSeer
### non-overlapping
```Python
$ cd FedIIH_2
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset CiteSeer --mode disjoint --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```
### overlapping
```Python
$ cd FedIIH_2
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset CiteSeer --mode overlapping --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```


## PubMed
### non-overlapping
```Python
$ cd FedIIH_4
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset PubMed --mode disjoint --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```
### overlapping
```Python
$ cd FedIIH_2
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset PubMed --mode overlapping --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```


## Amazon-Computer
### non-overlapping
```Python
$ cd FedIIH_6
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Computers --mode disjoint --frac 1.0 --n-rnds 200 --n-eps 3 --n-clients 10 --seed 42
```
### overlapping
```Python
$ cd FedIIH_4
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Computers --mode overlapping --frac 1.0 --n-rnds 200 --n-eps 3 --n-clients 10 --seed 42
```


## Amazon-Photo
### non-overlapping
```Python
$ cd FedIIH_6
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Photo --mode disjoint --frac 1.0 --n-rnds 200 --n-eps 2 --n-clients 10 --seed 42
```
### overlapping
```Python
$ cd FedIIH_10
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Photo --mode overlapping --frac 1.0 --n-rnds 200 --n-eps 2 --n-clients 10 --seed 42
```


## ogbn-arxiv
### non-overlapping
```Python
$ cd FedIIH_6
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset ogbn-arxiv --mode disjoint --frac 1.0 --n-rnds 200 --n-eps 2 --n-clients 10 --seed 42
```
### overlapping
```Python
$ cd FedIIH_6
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset ogbn-arxiv --mode overlapping --frac 1.0 --n-rnds 200 --n-eps 2 --n-clients 10 --seed 42
```


# Heterophilic datasets

## Roman-empire
### non-overlapping
```Python
$ cd FedIIH_4
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Roman-empire --mode disjoint --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```
### overlapping
```Python
$ cd FedIIH_4
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Roman-empire --mode overlapping --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```


## Amazon-ratings
### non-overlapping
```Python
$ cd FedIIH_4
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Amazon-ratings --mode disjoint --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```
### overlapping
```Python
$ cd FedIIH_4
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Amazon-ratings --mode overlapping --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```


## Minesweeper
### non-overlapping
```Python
$ cd FedIIH_6
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Minesweeper --mode disjoint --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```
### overlapping
```Python
$ cd FedIIH_4
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Minesweeper --mode overlapping --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```


## Tolokers
### non-overlapping
```Python
$ cd FedIIH_10
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Tolokers --mode disjoint --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```
### overlapping
```Python
$ cd FedIIH_10
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Tolokers --mode overlapping --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```


## Questions
### non-overlapping
```Python
$ cd FedIIH_8
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Questions --mode disjoint --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```
### overlapping
```Python
$ cd FedIIH_2
$ python main.py --gpu 0 --n-workers 1 --model fedhvae --dataset Questions --mode overlapping --frac 1.0 --n-rnds 100 --n-eps 1 --n-clients 10 --seed 42
```

## Citation
If you found our code or our paper useful in your work, please cite our work. Thank you very much!

```
@inproceedings{yu2025modeling,
  title={Modeling Inter-Intra Heterogeneity for Graph Federated Learning},
  author={Yu, Wentao and Chen, Shuo and Tong, Yongxin and Gu, Tianlong and Gong, Chen},
  booktitle={AAAI Conference on Artificial Intelligence},
  pages={1--8},
  year={2025}
}
```

## Reference
The code structure is inspired by the code in FED-PUB (https://github.com/JinheonBaek/FED-PUB).