from pyspark import SparkContext, SparkConf

DATA = "./data/*.txt"
OUTPUT_DIR = "counts" # name of the folder

def word_count():
    sc = SparkContext("local","Word count example")
    textFile = sc.textFile(DATA)
    counts = textFile.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
    counts.saveAsTextFile(OUTPUT_DIR)
    print("Number of partitions: ", textFile.getNumPartitions())
word_count()
