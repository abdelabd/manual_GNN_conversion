# manual_GNN_conversion

### to test compilation, prediction on sample data 
```bash
git clone https://github.com/abdelabd/manual_GNN_conversion
cd manual_GNN_conversion
python test_model.py test_config.yaml
```

### to test intermediate products
```bash
cd manual_GNN_conversion
python test_model_with_intermediates.py test_config.yaml
```

### for full testbench
```bash
cd manual_GNN_conversion
python full_test.py test_config.yaml
```
Assumes this version of hls4ml: https://github.com/abdelabd/hls4ml/tree/pyg_to_hls
