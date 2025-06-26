python Inference.py --instance 'SC' --solver 'TLNS' --destroy 'ml' --k 500 --kk 120
python Inference.py --instance 'CA' --solver 'TLNS' --destroy 'ml' --k 60000 --kk 3000
python Inference.py --instance 'MIS' --solver 'TLNS' --destroy 'ml' --k 30000 --kk 7000
python Inference.py --instance 'MVC' --solver 'TLNS' --destroy 'ml' --k 5500 --kk 1000

python Inference.py --instance 'SC' --solver 'TLNS' --destroy 'random' --k 8000 --kk 1600
python Inference.py --instance 'CA' --solver 'TLNS' --destroy 'random' --k 60000 --kk 3000
python Inference.py --instance 'MIS' --solver 'TLNS' --destroy 'random' --k 70000 --kk 7000
python Inference.py --instance 'MVC' --solver 'TLNS' --destroy 'random' --k 15000 --kk 1250

python Inference.py --instance 'SC' --solver 'LNS' --destroy 'ml' --k 175
python Inference.py --instance 'CA' --solver 'LNS' --destroy 'ml' --k 35000
python Inference.py --instance 'MIS' --solver 'LNS' --destroy 'ml' --k 12500
python Inference.py --instance 'MVC' --solver 'LNS' --destroy 'ml' --k 1250

python Inference.py --instance 'SC' --solver 'LNS' --destroy 'random' --k 4000
python Inference.py --instance 'CA' --solver 'LNS' --destroy 'random' --k 35000
python Inference.py --instance 'MIS' --solver 'LNS' --destroy 'random' --k 40000
python Inference.py --instance 'MVC' --solver 'LNS' --destroy 'random' --k 10000

python Inference.py --instance 'SC' --solver 'gurobi'
python Inference.py --instance 'CA' --solver 'gurobi'
python Inference.py --instance 'MIS' --solver 'gurobi'
python Inference.py --instance 'MVC' --solver 'gurobi'

python Inference.py --instance 'SC' --solver 'scip'
python Inference.py --instance 'CA' --solver 'scip'
python Inference.py --instance 'MIS' --solver 'scip'
python Inference.py --instance 'MVC' --solver 'scip'

