# Modeling Inter-Intra Heterogeneity for Graph Federated Learning

Official Code Repository for our paper - Modeling Inter-Intra Heterogeneity for Graph Federated Learning

**This is the first version of our codes, which seems to be not perfect. If the paper is accepted, we will publish the more elegant and concise codes on GitHub. We also promise to implement our method on the commercial federated learning frameworks, such as FATE, PaddleFL, and FederatedScope.**

## Requirement
- Python 3.9.16
- PyTorch 2.0.1
- PyTorch Geometric 2.3.0
- METIS (for data generation), https://github.com/james77777778/metis_python

## Data generation
Following command lines automatically to generate the dataset.
```sh
$ cd FedIIH_2/data/generators
$ python disjoint.py
$ python overlapping.py
```
or download from the Google Drive (https://drive.google.com/file/d/1RHziMtUg4fEXjdAK5Gqd9BrWwxos0l2e/view?usp=sharing) and then unzip it

## Parameter description


- `gpus`: specify gpus to use
- `num workers`: specify the number of workers on gpus (e.g. if your experiment uses 10 clients for every round then use less than or equal to 10 workers). The actual number of workers will be `num_workers` + 1 (one additional worker for a server).
- `FedIIH_2` means that the number of disentangled latent factors is set to 2 (K=2). Similarly, `FedIIH_10` means that the number of disentangled latent factors is set to 10 (K=10).

 

# Homophilic datasets
Following command lines to run the experiments.
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

## Computers
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


## Photo
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




