### Gather data

```bash
wget https://huggingface.co/datasets/sayakpaul/diode-subset-train/resolve/main/train_subset.tar.gz -O train_subset.tar.gz
wget http://diode-dataset.s3.amazonaws.com/val.tar.gz -O val.tar.gz
tar xf train_subset.tar.gz
tar xf val.tar.gz
```

### Installation

```bash
pip install -r requirements.txt
```

(Assumes a latest stable `torch` CUDA enabled environment)

### Authentication

```bash
huggingface-cli login
wandb login
```

### Running fine-tuning

```bash
python run_depth_estimation.py --head_init
```

Consult the other supported CLI arguments by running `python run_depth_estimation.py -h`.
