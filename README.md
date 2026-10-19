# HGR-CL

## Training

```bash
python train.py --method TEM --backbone CustomDecoupledHypergraphSGC --ILmode classIL --inter-task-edges false --minibatch false --dataset <dataset-name> --tem_args "budget:400;sampler:hyperedge_aware"
```

Requires Python 3.10 and CUDA-enabled PyTorch and DGL for training.

## Licensing

Public redistribution is pending license clarification. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); no repository-wide license has been assigned.

## Third-party notices

- **DGL-derived layers** (`Backbones/gnnconv.py`, `Backbones/gnns.py`): adapted from DGL; Apache 2.0 text is in `LICENSES/DGL-Apache-2.0.txt`.
- **HGNN reference** (`Backbones/standard_hgnn.py`): MIT license text is retained in `LICENSES/HGNN-MIT.txt` for any adapted material.
