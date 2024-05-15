docker run -v /home/ubuntu/DE300/lab6/word-count:/tmp/wc-demo -it \
	   -p 8888:8888 \
           --name wc-container \
	   pyspark-image
