# manual_GNN_conversion
```bash
git clone https://github.com/abdelabd/manual_GNN_conversion
cd manual_GNN_conversion
python test_model.py test_config.yaml
```

### to test intermediate products
- copy 'myproject_with_save.cpp' to the directory: hls_model.config.get_output_dir + '/firmware'
```bash
cd manual_GNN_conversion
python test_model_with_intermediates.py test_config.yaml
```
Assumes this version of hls4ml: https://github.com/abdelabd/hls4ml/tree/pyg_to_hls
