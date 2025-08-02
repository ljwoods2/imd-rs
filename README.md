
# Installation

First [install rustup](https://rustup.rs/)

```bash
git clone git@github.com:ljwoods2/imd-rs.git
cd imd-rs
conda env create --file devtools/conda-envs/test_env.yaml
conda activate imdclient-rs-test
maturin develop --features python
```

# Usage

```python
from quickstream import IMDClient

client = IMDClient("localhost", 8888, n_atoms, buffer_size=100*(1024**2))

client.get_imdframe()
client.stop()
```
