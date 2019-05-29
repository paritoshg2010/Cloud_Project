from flask import Flask, render_template, url_for, flash, redirect
from forms import PostReviewForm
import pandas as pd
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import *
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegressionModel
import pyspark
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType

conf = (pyspark.SparkConf().setAppName('FlaskApp').set("spark.executor.memory", "6g").setMaster("local[2]"))
conf.set("spark.driver.allowMultipleContexts", "true")
sc = pyspark.SparkContext(conf=conf)
sqlContext = SQLContext(sc)


pipelineFit = PipelineModel.load('pipelineFit_PRAA')
lrModel_final = LogisticRegressionModel.load('lrModel_final_PRAA')


                                         
app = Flask(__name__)

app.config['SECRET_KEY']='4ed6b7e2171cb2607487925fe0f320d'
WTF_CSRF_ENABLED = True


@app.route("/", methods=['GET', 'POST'])
def postreview():
    form = PostReviewForm()
    if form.validate_on_submit():
        if form.review.data is not None:
          Text = [form.review.data]
          df = pd.DataFrame({'Text':Text})
          df2 = sqlContext.createDataFrame(df)
          df2 = df2.dropna()
          text = "Text"
          target = "useful"
          tokenizer = Tokenizer(inputCol=text, outputCol="words")
          hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
          idf = IDF(inputCol='tf', outputCol="features")
          label_stringIdx = StringIndexer(inputCol = target, outputCol = "label")
          pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])
          transform_df = pipelineFit.transform(df2)
          predictions = lrModel_final.transform(transform_df)
          prediction_value= predictions.withColumn("value", predictions["prediction"].cast(IntegerType()))
          output = prediction_value.select('value').take(1)[0][0]

          if  output == 1:
             flash(f'Thanks for posting your review, this was really usefull..!', 'success')
             return redirect(url_for('postreview', _anchor='review_form'))
          else:
             flash('Thanks for posting, really appreciate if you can share more details..', 'danger')
             return redirect(url_for('postreview', _anchor='review_form'))
            
    return render_template('index.html', form=form)


if __name__ == '__main__':
  app.debug=True
  app.run(port=8081)

sc.stop()
