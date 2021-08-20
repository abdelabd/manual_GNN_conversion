# manual_GNN_conversion
- Download and extract data from here: https://www.dropbox.com/s/w20cy4x3w1q5zy7/processed_plus_pyg_small.tar.xz?dl=0
- Install this version of hls4ml: https://github.com/abdelabd/hls4ml/tree/pyg_to_hls_rebase (included in pyg_to_hls_env.yml)

To test the HLS implementation:
```bash
git clone https://github.com/abdelabd/manual_GNN_conversion
cd manual_GNN_conversion
python test_model.py test_config.yaml --n-graphs 100 --aggregation add --flow source_to_target --precision 'ap_fixed<16,8>' --max-nodes 28 --max-edges 51 --n-neurons 8
```

To sythesize the HLS implementation:
```bash
python test_model.py test_config.yaml --n-graphs 100 --aggregation add --flow source_to_target --precision 'ap_fixed<16,8>' --max-nodes 28 --max-edges 51 --n-neurons 8 --synth
```

