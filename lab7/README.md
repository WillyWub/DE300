# Lab 7 Assignment

## Machine learning with Spark

1 In the jupyter notebook 'ml_pyspark.ipynb', the PairwiseProduct step in the pipeline adds the cross product of any two numerical features to the dataframe. (e.g, if there are three variables a,b,c, then the PairwiseProduct step adds cross products a*b, a*c, b*c). In this part, you are required to also add the squared of numerical features to the dataframe (following the last example, you want to add a^2, b^2, c^2 to the dataframe). You should create a new Transofomer class for this purpose and add it into the pipeline.

2 The Cross Validation selects the maximum tree depth for the random forest model. There are many other hyper-parameters for the random forest. Search what they are, choose one such hyperparameter with its reasonble values and add to the ParameterGrid. 

3 Wrap all the functions you have into a python script (including the modification you have made to 1,2 above), and write a .sh file to execute the python script with EMR(spark on AWS, see the ppt tutorial in the folder spark-on-AWS). 

4 The python script should print out the new AUC ROC, Report and analyze what you find (compared to w.o modification of 1,2 above)


