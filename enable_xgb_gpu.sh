wget https://s3-us-west-2.amazonaws.com/xgboost-wheels/xgboost-0.81-py2.py3-none-manylinux1_x86_64.whl

pip uninstall -y xgboost

pip install xgboost-0.81-py2.py3-none-manylinux1_x86_64.whl
