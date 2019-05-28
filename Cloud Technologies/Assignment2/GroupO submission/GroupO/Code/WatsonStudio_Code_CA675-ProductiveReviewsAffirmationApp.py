
# coding: utf-8

# # Productive Reviews Affirmation App

# In the ever expanding world of e-commerce, an integral part of a product/service listed is the ‘Review’. Reviews, primarily given by people who have purchased a given product/service, forms a crucial part in the decision making process for a second person contemplating the purchase of the said product/service. A review strongly influences one to either purchase the product or decide against it. We have often seen how certain reviews are very helpful and some are not, hence, how can this mutually helpful process be enhanced?
# 
# This solution is a Machine Learning powered system that can determine if a 'Review' written on a website would prove to be helpful to others or not. The system can be compared to a password-strength indicator where the person is presented with information as to whether the password is strong or not. Similarly, this system can be deployed such that the end-user can be presented with information on the usefulness of the review he/she has entered.
# 
# The system will be trained with over 550,000 reviews scrapped from Amazon.

# In[1]:


#connectivity to IBM Cloud Object Storage
import ibmos2spark
# @hidden_cell
credentials = {
    'endpoint': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
    'service_id': 'iam-ServiceId-6f32cb78-577e-4dc3-a5da-4df27687941e',
    'iam_service_endpoint': 'https://iam.bluemix.net/oidc/token',
    'api_key': 'Ev5md1KqSpnLG3yNeeTuiDhvQEVICmgH1-9rBmYzhLXR'
}

configuration_name = 'os_cbd571eff81a4943b72c83ff53c338e6_configs'
cos = ibmos2spark.CloudObjectStorage(sc, credentials, configuration_name, 'bluemix_cos')

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
amz_df = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load(cos.url('Reviews.csv', 'ca675cloudtechnologies-donotdelete-pr-4raebf3cjvho6m'))


# # PySpark code to build Review-Helpfulness identifier (dev-test)

# In[2]:


#importing libraries
import pyspark.sql.functions as F
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# In[3]:


#determining useful or not
amz_df1 = amz_df.select(['HelpfulnessNumerator', 'HelpfulnessDenominator', 'Text'])
amz_df2 = amz_df1.withColumn('useful', F.when((amz_df1['HelpfulnessNumerator']/amz_df1['HelpfulnessDenominator'] >=0.5),1).otherwise(0)).drop('HelpfulnessNumerator').drop('HelpfulnessDenominator')


# In[4]:


amz_df2.show(5)


# In[5]:


#utils
amz_df2 = amz_df2.dropna()
text = "Text"
target = "useful"


# In[6]:


#dataset split
(train_set, test_set, val_set) = amz_df2.randomSplit([0.7, 0.15, 0.15], seed = 2000)


# In[7]:


train_set.show(10)


# In[8]:


##extracting features using TF-IDF
tokenizer = Tokenizer(inputCol=text, outputCol="words")
hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features")
label_stringIdx = StringIndexer(inputCol = target, outputCol = "label")
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])
pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
val_df = pipelineFit.transform(val_set)


# In[9]:


train_df.show(5)
val_df.show(5)


# In[10]:


#Model 1 - building logistic regression ML model
lr = LogisticRegression(maxIter=100)
lrModel = lr.fit(train_df)
predictions = lrModel.transform(val_df)


# In[11]:


#evaluation
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print(evaluator.getMetricName())
evaluator.evaluate(predictions)


# In[12]:


#evaluation
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())
accuracy


# In[13]:


#Model 2 - N-gram Implementation
from pyspark.ml.feature import NGram, VectorAssembler
from pyspark.ml.feature import ChiSqSelector
def build_ngrams_wocs(inputCol=["text","target"], n=3):
    tokenizer = [Tokenizer(inputCol=text, outputCol="words")]
    ngrams = [
        NGram(n=i, inputCol="words", outputCol="{0}_grams".format(i))
        for i in range(1, n + 1)
    ]

    cv = [
        CountVectorizer(vocabSize=5460,inputCol="{0}_grams".format(i),
            outputCol="{0}_tf".format(i))
        for i in range(1, n + 1)
    ]
    idf = [IDF(inputCol="{0}_tf".format(i), outputCol="{0}_tfidf".format(i), minDocFreq=5) for i in range(1, n + 1)]

    assembler = [VectorAssembler(
        inputCols=["{0}_tfidf".format(i) for i in range(1, n + 1)],
        outputCol="features"
    )]
    label_stringIdx = [StringIndexer(inputCol = target, outputCol = "label")]
    lr = [LogisticRegression(maxIter=100)]
    return Pipeline(stages=tokenizer + ngrams + cv + idf+ assembler + label_stringIdx+lr)


# In[14]:


trigramwocs_pipelineFit = build_ngrams_wocs().fit(train_set)
predictions_wocs = trigramwocs_pipelineFit.transform(val_set)


# In[15]:


accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())
roc_auc = evaluator.evaluate(predictions)


# In[16]:


print("Accuracy Score: {0:.4f}".format(accuracy))
print("ROC-AUC: {0:.4f}".format(roc_auc))


# Testing test data with logistic since ROC was similar to N-gram and took much lesser amount of time

# In[17]:


#Testing with test data
test_df = pipelineFit.transform(test_set)
predictions = lrModel.transform(test_df)


# In[18]:


#evaluating the models performance on the test data
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)


# ## PySpark code to build Review-Helpfulness identifier (final build)

# In[19]:


#transforming all the records
tokenizer = Tokenizer(inputCol=text, outputCol="words")
hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features")
label_stringIdx = StringIndexer(inputCol = target, outputCol = "label")
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])
pipelineFit = pipeline.fit(amz_df2)
final_df = pipelineFit.transform(amz_df2)
final_df.show(5)


# In[20]:


#Saving the Pipeline model
from pyspark.ml import PipelineModel
pipelineFit.save(cos.url('pipelineFit_PRAA', 'ca675cloudtechnologies-donotdelete-pr-4raebf3cjvho6m'))


# In[21]:


#training the regression model with full dataset
lr_final = LogisticRegression(maxIter=100)
lrModel_final = lr_final.fit(final_df)


# In[22]:


#saving the final model
lrModel_final.save(cos.url('lrModel_final_PRAA', 'ca675cloudtechnologies-donotdelete-pr-4raebf3cjvho6m'))


# # Testing the model with new data

# In[2]:


#connectivity to object storage
import ibmos2spark
credentials = {
    'endpoint': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
    'service_id': 'iam-ServiceId-6f32cb78-577e-4dc3-a5da-4df27687941e',
    'iam_service_endpoint': 'https://iam.bluemix.net/oidc/token',
    'api_key': 'Ev5md1KqSpnLG3yNeeTuiDhvQEVICmgH1-9rBmYzhLXR'
}

configuration_name = 'os_cbd571eff81a4943b72c83ff53c338e6_configs'
cos = ibmos2spark.CloudObjectStorage(sc, credentials, configuration_name, 'bluemix_cos')


# In[3]:


#importing libraries
import pyspark.sql.functions as F
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegressionModel


# In[24]:


#to clear input variable
del(input)


# In[25]:


#on screen input
try:
    input = input('Please enter the product review - ')
except NameError: 
    pass


# In[26]:


#converting the data into a dataframe
import pandas as pd
Text = [input]
df = pd.DataFrame({'Text':Text})
df2 = spark.createDataFrame(df)
df2 = df2.dropna()


# In[27]:


#loading the saved pipeline model
pipelineFit = PipelineModel.load(cos.url('pipelineFit_PRAA', 'ca675cloudtechnologies-donotdelete-pr-4raebf3cjvho6m'))


# In[28]:


#transforming the data
text = "Text"
target = "useful"
tokenizer = Tokenizer(inputCol=text, outputCol="words")
hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features")
label_stringIdx = StringIndexer(inputCol = target, outputCol = "label")
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])
transform_df = pipelineFit.transform(df2)


# In[29]:


#loading saved the ML model
lrModel_final = LogisticRegressionModel.load(cos.url('lrModel_final_PRAA', 'ca675cloudtechnologies-donotdelete-pr-4raebf3cjvho6m'))


# In[30]:


#predicting if the review is helpful
predictions = lrModel_final.transform(transform_df)
predictions.show()

