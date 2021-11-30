#!/bin/bash

# ROC curves
python make_roc.py test_config.yaml --max-nodes 113 --max-edges 196 --n-graphs 1000
python make_roc.py test_config_1GeV.yaml --max-nodes 162 --max-edges 326 --n-graphs 1000

# extract scan data
python extract_scan_data.py --dir numbers_for_paper/n28xe56_pipeline_reports
python extract_scan_data.py --dir numbers_for_paper/n448xe896_rf1_dataflow_reports
python extract_scan_data.py --dir numbers_for_paper/n448xe896_rf8_dataflow_reports
python extract_scan_data.py --dir numbers_for_paper/size_scan_rf1_reports
python extract_scan_data.py --dir numbers_for_paper/size_scan_rf8_reports

# plot scans
python plot_scans.py --reuse 8 --precision 'ap_fixed<14,7>' --dir numbers_for_paper/n28xe56_pipeline_reports --max-nodes 28 --max-edges 56 --paradigm pipeline
python plot_scans.py --reuse 1 --precision 'ap_fixed<14,7>' --dir numbers_for_paper/n448xe896_rf1_dataflow_reports --max-nodes 448 --max-edges 896 --paradigm dataflow
python plot_scans.py --reuse 8 --precision 'ap_fixed<14,7>' --dir numbers_for_paper/n448xe896_rf8_dataflow_reports --max-nodes 448 --max-edges 896 --paradigm dataflow

# plot size scans
python plot_size_scans.py --reuse 1 --precision 'ap_fixed<14,7>' --dir numbers_for_paper/size_scan_rf1_reports
python plot_size_scans.py --reuse 8 --precision 'ap_fixed<14,7>' --dir numbers_for_paper/size_scan_rf8_reports


