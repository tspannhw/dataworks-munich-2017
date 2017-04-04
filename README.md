# dataworks-munich-2017
This is the demo of [Revolutionize Text Mining with Spark and Zeppelin](https://dataworkssummit.com/munich-2017/sessions/revolutionize-text-mining-with-spark-and-zeppelin/) at DataWorks Summit Munich 2017.

The goal of this demo is to explore some of the main Spark MLlib features on a single practical task:
analysing a collection of text documents (newsgroups posts) on twenty different topics.
In this section we will see how to:
* load the file contents and the categories
* extract feature vectors suitable for machine learning
* train a linear model to perform categorization
* use a grid search strategy to find a good configuration of both the feature extraction components and the classifier

Actually this is more or less the translation of the scikit-learn example:
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
to illustrate how to use Spark MLlib to do similar work but with large scale dataset.

Note: Before run this example, please get into `./data` and run `python fetch_data.py`, then you will get the traing and test dataset.