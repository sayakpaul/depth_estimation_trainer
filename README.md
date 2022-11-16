### Gather data

**Ensure you're within this project root.**

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
python run_depth_estimation.py --head_init --log_code
```

Consult the other supported CLI arguments by running `python run_depth_estimation.py -h`.

### Misc

The script is integrated with Weights and Biases (WandB) which
can [automatically keep track](https://docs.wandb.ai/ref/app/features/panels/code) of
the Git state of the project. So, it's recommended to first create a branch if there
are any code changes, commit the changes to the branch, and then launch the experiment. This way
we can easily track the changes from the WandB console. 
