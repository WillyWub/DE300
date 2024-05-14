







docker run -v /Users/robertsu/Documents/TA/DE300/lab_doc/lab6/word-count:/tmp/wc-demo -it \
	   -p 8888:8888 \
           --name wc-container \
	   pyspark-image
