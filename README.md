Download Link: https://assignmentchef.com/product/solved-cs434-assignment-2-naive-bayes-multinomial-for-sentiment-analysis
<br>
In this assignment you will implement the Naive Bayes (Multinomial) classifier for Sentiment Analysis of an IMDB movie review dataset (a highly polarized dataset with 50000 movie reviews). The primary task is to classify the reviews into negative and positive.

More about the dataset: <a href="http://ai.stanford.edu/~amaas/data/sentiment/">http://ai.stanford.edu/˜amaas/data/sentiment/</a>

For the Multinomial model, each document is represented by a vector of integer-valued variables, i.e., <strong>x </strong>= (<em>x</em><sub>1</sub><em>,x</em><sub>2</sub><em>,…,x</em><sub>|<em>V </em>|</sub>)<em><sup>T </sup></em>and each variable <em>x<sub>i </sub></em>corresponds to the <em>i</em>-th word in a vocabulary <em>V </em>and represents the number of times it appears in the document. The probability of observing a document <strong>x </strong>given its class label <em>y </em>is defined as (for example, for <em>y </em>= 1):

|<em>V </em>| <em>p</em>(<strong>x</strong>|<em>y </em>= 1) = <sup>Y</sup><em>P</em>(<em>w<sub>i</sub></em>|<em>y </em>= 1)<em><sup>x</sup></em><em><sup>i</sup></em>

<em>i</em>=1

Here we assume that given the class label <em>y</em>, each word in the document follows a multinomial distribution of |<em>V </em>| outcomes and <em>P</em>(<em>w<sub>i</sub></em>|<em>y </em>= 1) is the probability that a randomly selected word is word <em>i </em>for a document of the positive class. Note that <sup>P</sup><em><sub>i</sub></em><sub>=1 </sub><em>P</em>(<em>w<sub>i</sub></em>|<em>y</em>) = 1 for <em>y </em>= 0 and <em>y </em>= 1. Your implementation need to estimate <em>p</em>(<em>y</em>), and <em>P</em>(<em>w<sub>i</sub></em>|<em>y</em>) for <em>i </em>= 1<em>,</em>···<em>,</em>|<em>V </em>|, and <em>y </em>= 1<em>,</em>0 for the model. For <em>p</em>(<em>y</em>), you can use the MLE estimation. For <em>P</em>(<em>w<sub>i</sub></em>|<em>y</em>), you MUST use Laplace smoothing for the model. One useful thing to note is that when calculating the probability of observing a document given its class label, i.e., <em>p</em>(<strong>x</strong>|<em>y</em>), it can and will become overly small because it is the product of many probabilities. As a result, you will run into underflow issues. To avoid this problem, your implementation should operate with log of the probabilities.

<h1>1           Description of the dataset</h1>

The data set provided are in two parts:

<ul>

 <li>csv: This contains a single column called Reviews where each row contains a movies review. There are total of 50K rows. The first 30K rows should be used as your Training set (to train your model). The next 10K should be used as the validation set (use this for parameter tuning). And the last 10K rows should be used as the test set (predict the labels).</li>

 <li>IMDB labels.csv: This contains 40K labels. Please use the first 30K labels for the training data and the last 10K labels for validation data. The labels for test data is not provided, we will use that to evaluate your predicted labels.</li>

</ul>

<h1>2           Data cleaning and generating BOW representation</h1>

<strong>Data Cleaning. </strong>Pre-processing is need to makes the texts cleaner and easy to process. The reviews columns are comments provided by users about the movie. These are known as ”dirty text” that required further cleaning. Typical cleaning steps include a) Removing html tags; b) Removing special characters; c) Converting text to lower case d) replacing punctuation characters with spaces; and d) removing stopwords i .e . articles, pronouns from consideration. You will not need to implement these functionalities and we will provide some starter code containing these functions for you to use.

<strong>Generating BOW representation. </strong>To transform from variable length reviews to fixed-length vectors, we use the Bag Of Words technique. It uses a list of words called ”vocabulary”, so that given an input text we can output a vector of word counts for each word in the vocabulary. You can use the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html">CountVectorizer </a>functionality from sklearnstarter to go over the full 50K reviews to generate the vocabulary and create the feature vectors representing each review. Not that the CountVectorizer function has several tunable parameters that can directly impact the result feature representation. This includes <em>max features </em>:, which specifies the maximum number of features (by considering terms with high frequency); <em>max df </em>and <em>min df</em>, which filter the words from the dictionary if its document frequency is too high (<em>&gt; max df</em>) or too low (<em>&lt; min df</em>) respectively.

<h1>3           What you need to do</h1>

<ol>

 <li> Apply the above described data cleaning and feature generation steps to generate the BOW representation for all 50k reviews. For this step, we will use the default value for <em>max df </em>and <em>min df </em>and set <em>max features </em>= 2000.</li>

 <li>Train a multi-nomial Naive Bayes classifier with Laplace smooth with <em>α </em>= 1 on the training set. This involves learning <em>P</em>(<em>y </em>= 1)<em>,P</em>(<em>y </em>= 0), <em>P</em>(<em>w<sub>i</sub></em>|<em>y </em>= 1) for <em>i </em>= 1<em>,…,</em>|<em>V </em>| and <em>P</em>(<em>w<sub>i</sub></em>|<em>y </em>= 0) for <em>i </em>= 1<em>,…,</em>|<em>V </em>| from the training data (the first 30k reviews and their associated labels).</li>

 <li> Apply the learned Naive Bayes model to the validation set (the next 10k reviews) and report the validation accuracy of the your model. Apply the same model to the testing data and output the predictions in a file, which should contain a single column of 10k labels (0 (negative) or 1 (positive)). Please name the file test-prediction1.csv.</li>

 <li>) Tuning smoothing parameter <em>alpha</em>. Train the Naive Bayes classifier with different values of <em>α </em>between 0 to 2 (incrementing by 0.2). For each <em>alpah </em>value, apply the resulting model to the validation data to make predictions and measure the prediction accuracy. Report the results by creating a plot</li>

</ol>

with value of <em>α </em>on the <em>x</em>-axis and the validation accuracy on the <em>y</em>-axis. <u>Comment </u>on how the validation accuracy change as <em>α </em>changes and provide a short explanation for your observation. Identify

the best <em>α </em>value based on your experiments and apply the corresponding model to the test data and output the predictions in a file, named test-prediction2.csv.

<ol start="5">

 <li><strong>Tune your heart out. </strong>For the last part, you are required to tune the parameters for the CountVectorizer (<em>max feature</em>, <em>max df</em>, and <em>min df</em>). You can freely choose the range of values you wish<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> to test for these parameters and use the validation set to select the best model. Please <u>describe </u>your strategy for choosing the value ranges and report the best parameters (as measured by the prediction accuracy on the validation set) and the resulting model’s validation accuracy. You are also required to apply the chosen best model to make predictions for the testing data, and output the predictions in a file, named test-prediction3.csv.</li>

</ol>

<a href="#_ftnref1" name="_ftn1">[1]</a> You are encouraged to try your best to tune these parameters. Higher validation accuracy and testing accuracy will be rewarded with possible bonus points.