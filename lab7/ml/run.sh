docker run -v /home/ubuntu/DE300/lab7/ml:/tmp/ml -it \
           -p 8888:8888 \
           --name spark-sql-container \
	   pyspark-image
