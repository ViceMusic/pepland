# PepLand Inference

Generate peptide embeddings using the pretrained PepLand model.

## Quick Start

### 1. Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate pepland-inference
```

### 2. Prepare Input Data

Create a `.smi` file with one SMILES string per line:

```
OC(=O)CC[C@@H](C(=O)N[C@@H](Cc1ccccc1)C...
[NH3+]CCCC[C@@H](C(=O)N[C@H](C(=O)N[C@H]...
```

### 3. Configure Inference

Edit `../configs/inference.yaml`:

```yaml
mode:
  ddp: false

inference:
  device_ids: [0]              # GPU device IDs, empty for CPU
  data: '../data/example.smi'  # Path to input SMILES file
  model_path: "./cpkt/model"   # Path to model checkpoint
  pool: avg                    # Pooling method: avg or max
  atom_index: false            # false for peptide embedding, or index for atom embedding
```

### 4. Run Inference

```bash
cd /home/richard/projects/pepland/inference
python inference_pepland.py
```

## Output

- **Peptide Embedding**: Shape `(N, 300)` where N is the number of input SMILES
- **Atom Embedding**: If `atom_index` is set, returns embedding for specific atom position

## Environment Details

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.8 | Runtime |
| PyTorch | 1.11.0+cu113 | Deep Learning |
| DGL | 0.9.1 (CUDA 11.3) | Graph Neural Networks |
| RDKit | 2024.x | Molecule Processing |
| MLflow | 1.30.0 | Model Loading |
| OmegaConf | 2.2.x | Configuration |
| scikit-learn | 1.3.x | Model utilities |

## File Structure

```
inference/
├── environment.yaml       # Conda environment file
├── README.md              # This file
├── inference_pepland.py   # Main inference script
├── process.py             # Molecule to graph conversion
├── tokenizer/             # Peptide tokenization
│   ├── pep2fragments.py
│   └── vocabs/
│       └── Vocab_SIZE258.txt
└── cpkt/
    └── model/             # Pretrained model checkpoint
```

## Usage Example

```python
import torch
import dgl
from omegaconf import OmegaConf

# Import from inference directory
from process import Mol2HeteroGraph
from inference_pepland import load_model

# Load config and model
cfg = OmegaConf.load('../configs/inference.yaml')
model = load_model(cfg)
model.eval()

# Process SMILES
smiles = "CC(C)C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)C)C(=O)O"
graph = Mol2HeteroGraph(smiles)
bg = dgl.batch([graph])

# Get embeddings
with torch.no_grad():
    atom_embed, frag_embed = model(bg)
    print(f"Atom embedding shape: {atom_embed.shape}")      # (num_atoms, 300)
    print(f"Fragment embedding shape: {frag_embed.shape}")  # (num_fragments, 300)
```

## Batch Processing

```python
import torch
import dgl
import torch.nn as nn
from omegaconf import OmegaConf
from process import Mol2HeteroGraph
from inference_pepland import load_model, split_batch

# Load model
cfg = OmegaConf.load('../configs/inference.yaml')
model = load_model(cfg)
model.eval()
device = torch.device("cpu")

# Process multiple SMILES
smiles_list = [
    "CC(C)C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)C)C(=O)O",
    "CC[C@H](C)[C@H](NC(=O)CNC(=O)[C@H](CC(C)C)NC(=O)C)C(=O)O"
]

graphs = [Mol2HeteroGraph(smi) for smi in smiles_list]
bg = dgl.batch(graphs)

# Get embeddings with pooling
with torch.no_grad():
    atom_embed, frag_embed = model(bg)
    bg.nodes['a'].data['h'] = atom_embed
    bg.nodes['p'].data['h'] = frag_embed
    
    atom_rep = split_batch(bg, 'a', 'h', device)
    frag_rep = split_batch(bg, 'p', 'h', device)
    
    # Average pooling
    pool = nn.Sequential(
        nn.AdaptiveAvgPool1d(output_size=1),
    )
    pep_embeds = pool(torch.cat([atom_rep, frag_rep], dim=1).permute(0, 2, 1))
    pep_embeds = pep_embeds.squeeze(-1).numpy()
    
    print(f"Peptide embeddings shape: {pep_embeds.shape}")  # (2, 300)
```

## Notes

- CUDA 11.3 compatible GPU required for GPU acceleration
- For CPU-only inference, set `device_ids: []` in config
- Model outputs 300-dimensional embeddings
- The warning about PyTorch version mismatch (1.11.0 vs 1.11.0+cu113) can be safely ignored
