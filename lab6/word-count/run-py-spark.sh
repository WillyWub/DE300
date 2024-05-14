# export PYSPARK_DRIVER_PYTHON=python3 only in cluster mode
/bin/rm -r -f counts

export PYSPARK_PYTHON=../demos/bin/python3
/opt/spark/bin/spark-submit --archives ../demos.tar.gz#demos wc.py

