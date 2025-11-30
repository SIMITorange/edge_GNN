edge_GNN surrogate project
==========================

What this project does
----------------------
- Trains a graph neural network surrogate for semiconductor device fields.
- Inputs (per node): x, y, doping, Vds (broadcast), plus optional Fourier-lifted coordinates.
- Outputs (per node): ElectrostaticPotential, ElectricField_x, ElectricField_y, SpaceCharge.
- Data source: `meshgraph_data.h5` produced by the provided Delaunay-based preprocessing script.

Quick start
-----------
1) Install deps (adjust to your CUDA/PyG build):
```
pip install -r requirements.txt
```
2) Update `config.py` paths if your HDF5 lives elsewhere.
3) Train (GPU default; add `--device cpu` to force CPU and AMP will auto-disable):
```
python train.py --data D:\...\meshgraph_data.h5
python train.py --data D:\...\meshgraph_data.h5 --device cpu
```
   - Checkpoints + normalization saved under `artifacts/`.
   - TensorBoard logs under `logs/`.
4) Inference on a specific device/sheet:
```
python inference.py --group n1 --sheet 0 --checkpoint artifacts/edge_gnn.ckpt --norm artifacts/normalization.json
python inference.py --group n1 --sheet 0 --checkpoint artifacts/edge_gnn.ckpt --norm artifacts/normalization.json --device cpu
```
   - Predictions saved to `outputs/inference.npy`.
5) Visualize losses and prediction vs. truth:
```
python tools/visualize.py --metrics artifacts/metrics.pkl --pred outputs/inference.npy
```

Project layout
--------------
- `config.py`: defaults for paths, data/normalization, model, loss, training hyperparams.
- `src/data/`: HDF5 loader, Fourier features, normalization fitting/persistence.
- `src/models/`: residual message-passing backbone + multi-head decoders.
- `src/training/`: composite loss and Trainer (AMP, early stop, checkpoints).
- `train.py`: end-to-end training entry.
- `inference.py`: loads checkpoint + normalization for fast predictions.
- `tools/visualize.py`: plots loss curves, field comparisons, model summary.
- `artifacts/`, `logs/`, `outputs/`: created automatically for checkpoints, TB logs, and plots.

Normalization and loss design
-----------------------------
- Coordinates: min-max per axis.
- Doping: signed log1p + standardization (handles wide dynamic range).
- Vds: standardization on the per-sheet max potential.
- Targets: potential uses standardization; E-fields and space charge use robust scaling (median/IQR).
- Loss = SmoothL1 + relative-L1 + edge total-variation + optional potential↔E consistency.

Notes on Vds and sheets
-----------------------
- Each HDF5 group (e.g., `n12`) may contain multiple sheets (bias points). The dataset treats each sheet as one sample.
- Vds is automatically computed as the max electrostatic potential of that sheet and broadcast to nodes as an input feature.
- Fourier mapping (`num_features=8`, `sigma=1.0` by default) is applied on normalized coordinates to help fit sharp field jumps/edge effects.

Tips
----
- Start with batch_size=1 for large meshes (~100k nodes). Increase workers if memory allows.
- If gradients explode, lower `lr` or `grad_clip` in `config.py`.
- To emphasize a specific output, adjust weights in `LossConfig` inside `config.py`.
