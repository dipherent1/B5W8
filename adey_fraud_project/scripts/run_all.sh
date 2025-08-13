#!/usr/bin/env bash
set -euo pipefail
python -m src.train_ecommerce --data_dir data --out_dir reports/ecommerce
python -m src.train_creditcard --data_dir data --out_dir reports/creditcard
