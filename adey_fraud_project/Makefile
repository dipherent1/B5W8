install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

ecommerce:
	python -m src.train_ecommerce --data_dir data --out_dir reports/ecommerce

creditcard:
	python -m src.train_creditcard --data_dir data --out_dir reports/creditcard
