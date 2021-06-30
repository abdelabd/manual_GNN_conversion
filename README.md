# manual_GNN_conversion
- Download and extract data from here: https://www.dropbox.com/s/w20cy4x3w1q5zy7/processed_plus_pyg_small.tar.xz?dl=0
- Install this version of hls4ml: https://github.com/abdelabd/hls4ml/tree/pyg_to_hls_rebase (included in pyg_to_hls_env.yml)

### To test HLS implementation
```bash
git clone https://github.com/abdelabd/manual_GNN_conversion
cd manual_GNN_conversion
python test_model.py test_config.yaml --n-graphs=100 --aggregation-method all --flow all --save-intermediates
```

### To test HLS logic in Python
```bash
git clone https://github.com/abdelabd/manual_GNN_conversion
cd manual_GNN_conversion
python test_wrappers.py test_config.yaml --n-graphs=100 --aggregation-method all --flow all --complexity all
```
