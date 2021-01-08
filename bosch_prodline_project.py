from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("boschdata").getOrCreate()

#read the data
raw_data = spark.read.format("csv").option("header","true").option("inferSchema","true").load(r"/home/iitp/train_numeric17.csv")
print(" ")
print(raw_data.columns)
print(" ")
print((raw_data.count(), len(raw_data.columns)))
print(" ")
print(raw_data.select("Response").show())
print(" ")
from pyspark.sql.functions import isnan, when, count, col,isnull
#print("nan values")
#print(raw_data.select([count(when(isnan(c), c)).alias(c) for c in raw_data.columns]).show())
print("   ")
print("null values")
print(raw_data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in raw_data.columns]).show())
print("   ")

raw_data = raw_data.fillna(0)

#raw_data.drop('Id').collect()
#from pyspark.sql.types import DoubleType
#for c in raw_data.columns:
    # add condition for the cols to be type cast
 #   raw_data = raw_data.withColumn(c, raw_data[c].cast('double'))

#from pyspark.ml.feature import Imputer

#impute = Imputer(
 #   inputCols=raw_data.columns,
  #  outputCols=["{}_imputed".format(c) for c in raw_data.columns]
#)

#model = impute.fit(raw_data)
#raw_data = model.transform(raw_data)

print("after filling nan ")
#print(raw_data.show(5))
print(" ")

#print(raw_data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in raw_data.columns]).show())
print(" ")

cols = raw_data.columns
cols.remove("Response")
cols.remove("Id")

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=cols,outputCol="features")
raw_data = assembler.transform(raw_data)
print(" ")
print(raw_data.select("features").show(truncate=False))
print(" ")



print("PCA")

from pyspark.ml.feature import PCA
pca = PCA(k=40, inputCol="features", outputCol="PCAFeatures")
model = pca.fit(raw_data)

raw_data1 = model.transform(raw_data).select("PCAFeatures")
print("\n Data Model After Applying PCA \n")
print(raw_data1.show(truncate=False))
print(" ")

#Scaling the data
from pyspark.ml.feature import StandardScaler
standardscaler = StandardScaler().setInputCol("PCAFeatures").setOutputCol("Scaled_features")
raw_data_scaled = standardscaler.fit(raw_data1).transform(raw_data1)
print("\n Data After Scaling \n ")
print(raw_data_scaled.select("PCAFeatures","Scaled_features").show(5))
print(" ")



#Applying KMeans Clustering
print("Applying KMeans")
from pyspark.ml.clustering import KMeans
# from pyspark.ml.evaluation import ClusteringEvaluator


# Trains a k-means model.
kmeans = KMeans().setK(3).setSeed(1)
model = kmeans.fit(raw_data_scaled)

# Make predictions
predictions = model.transform(raw_data_scaled)

# # Evaluate clustering by computing Silhouette score
# evaluator = ClusteringEvaluator()

# silhouette = evaluator.evaluate(predictions)
# print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)





