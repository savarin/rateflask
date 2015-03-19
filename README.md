rateflask
========

rateflask predicts the rate of return at inception of a Lending Club loan, web 
version at [rateflask.com][rateflask.com].

### Description
Analysis of Lending Club data tends to focus on loans that have 
[already matured][already matured]. Matured loans, however, comprise 
[less than 10%][less than 10%] of loans issued. This conundrum inspired a new 
methodology to enable loan comparisons be made, regardless as to whether the 
loan has matured, is ongoing or yet to be issued.

The methodology involves using Random Forests to predict the expected loan 
payment for a particular month, then aggregated across the whole period to give 
a single rate of return metric for each loan. This allows 90% of loans issued be 
used as training data, and the remaining set of matured loans as validation.

![][issuance]

### Details

The model consists of 4 x 36 individual [Random Forests][Random Forests] 
sub-models, one for each grade-month pair (grades A - D, in the period Jan 2012 
- Dec 2014). The training data is the set of 3-year loans issued between 2012 
and 2014, i.e. ongoing loans. Loan details (FICO, monthly income, etc.) are used 
as features, and the loan status (current, in default, etc) as targets.

The [loan status][loan status] is used to calculate the probability of each 
payment made, and aggregated to give the rate of return of that loan. Viewed as 
a black box, the model takes in loan features as input, and outputs the expected 
rate of return.

The trained model is validated against 3-year loans issued between 2009 and
2011, i.e. loans that have matured. The validation process involves comparing 
the actual rate of return, calculated purely with actual payment data, against
the expected rate of return, calculated purely on loan features. The actual and 
predicted rate of return are illustrated in the graph below by blue and green
respectively.

![][compare]

The graph below shows the improvement in rate of return with an active selection
strategy based on the model, compared to choosing a loan of a specified sub-
grade at random. The active selection strategy involves using the model to
generate the predicted rate of return, ranking the loans and identifying the top
quartile. Details on how the chart is generated can be found [here][charts].

![][quartile]

For further details:
* [Mechanics: How does the model work?][mechanics]
* [Validation: Does the model work?][validation]
* [Application: Why do I care?][application]

Presentation slides can be found [here][presentation]. The charts were generated
with R's ggplot via rpy2, and details can be found [here][charts].

### Requirements
* numpy 1.9.0
* scipy 0.14.0
* pandas 0.14.1
* scikit-learn 0.14.1
* matplotlib 1.3.1
* flask 0.10.1
* lendingclub 0.1.8
* pymongo 2.7.2
* psycopg2 2.5.3
* dill 0.2.2

### Installation
1. Clone this repo.
2. Download the full version data (~270 MB) from the 
[Lending Club website][Lending Club website] or from the following 
[Dropbox address][Dropbox address], and place in a directory labeled `data`.
3. Install the listed requirements.
4. (Optional) Start up a MongoDB instance, and a PostgreSQL database named 
'rateflask'.

To run the production version locally, run `python app.py` (or `sudo python 
app.py` should there be permission errors) in terminal from the repo directory.
Direct your browser to `0.0.0.0:8000` to view the app, and to 
`0.0.0.0:8000/refresh` to update the data (requires Lending Club login). For 
debugging, run `python app.py debug`.

To test if the installation has been successful, run `python test.py` from the 
same location. To run the model against the validation set, run `python test.py 
compare`. Please note that the validation process might take some time.

### Modules

**model - rate of return prediction and validation**
* model.py - core prediction model, trained on 2012-14 loan data
* validation.py - validates prediction model with 2009-11 loan data
* start - trains new model on first start

**helpers - data processing and cashflow generation**
* preprocessing.py - cleans up data and fills missing values
* postprocessing.py - creates files for charts and data table
* cashflow.py - generates cashflows and compounding curves

**transfers - file input/output, API requests and database insertions**
* fileio.py - dumping and loading data with pickle/dill
* retrieve.py - requests data from Lending Club API
* database.py - inserts data to MongoDB and PostgreSQL

### Next steps

Portfolio selection model that selects the highest-returning diversified 
portfolio based on a user's desired risk profile.

### License

Copyright (c) 2015 Rateflask

Licensed under the MIT licence.


<!-- links -->

[rateflask.com]: http://www.rateflask.com

[already matured]: https://www.lendingrobot.com/#/resources/charts
[less than 10%]: https://www.lendingclub.com/info/statistics.action
[issuance]: static/images/issuance.png

[Random Forests]: http://en.wikipedia.org/wiki/Random_forest
[loan status]: https://www.lendingclub.com/info/demand-and-credit-profile.action
[issuance]: static/images/compare.png
[charts]: http://nbviewer.ipython.org/github/savarin/rateflask/blob/master/notebooks/charts.ipynb
[quartile]: static/images/quartile.png

[mechanics]: http://nbviewer.ipython.org/github/savarin/rateflask/blob/master/notebooks/mechanics.ipynb
[validation]: http://nbviewer.ipython.org/github/savarin/rateflask/blob/master/notebooks/validation.ipynb
[application]: http://nbviewer.ipython.org/github/savarin/rateflask/blob/master/notebooks/application.ipynb
[presentation]: https://github.com/savarin/rateflask/blob/master/notebooks/presentation.pdf

[Lending Club website]: https://www.lendingclub.com/info/download-data.action
[Dropbox address]: https://www.dropbox.com/sh/pmwh81xl7bi5axv/AABSewOpldF2zdqr6JOP5lNha?dl=0