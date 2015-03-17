rateflask
========

rateflask predicts the rate of return at inception of a Lending Club loan, web 
version at [rateflask.com](http://www.rateflask.com).

### Description
Analysis of Lending Club data tends to focus on loans that have already matured. 
Matured loans, however, comprise less than 10% of loans issued. This conundrum 
inspired a new methodology to enable loan comparisons be made, regardless as to 
whether the loan has matured, is ongoing or yet to be issued.

The methodology involves using Random Forests to predict the expected loan 
payment for a particular month, then aggregated across the whole period to give 
a single rate of return metric. This allows 90% of loans issued be used as 
training data, and the set of matured loans as validation.

### Details

The model consists of 4 x 36 individual Random Forest sub-models, one for each
grade (A, B, C and D) - month pair (Jan 2012 - Dec 2014). The training data is 
the set of 3-year loans issued between 2012 and 2014, i.e. ongoing loans. Loan
details (FICO, monthly income, etc.) are used as features, and the loan status
(current, in default, etc) as targets.

The loan status is used to calculate the probability of each payment made, and 
aggregated to give the rate of return of that loan. As a black box, the model 
takes in loan features as input, and outputs the expected rate of return.

The trained model is validated against 3-year loans issued between 2009 and
2011, i.e. loans that have matured. The validation process involves comparing 
the actual rate of return, calculated purely with actual payment data, against
the expected rate of return, calculated purely on loan features.

For further details:
* [Validation: Does the model work?](http://nbviewer.ipython.org/github/savarin/rateflask/blob/master/notebooks/validation.ipynb)

To be added:
* Mechanics: How does the model work?
* Application: Why do I care?

Presentation slides can be found [here](https://github.com/savarin/rateflask/blob/master/presentation.pdf), with details on how the charts and 
figures are generated [here](http://nbviewer.ipython.org/github/savarin/rateflask/blob/master/notebooks/charts.ipynb).

### Requirements
* numpy 1.9.0
* pandas 0.14.1
* scikit-learn 0.14.1
* flask 0.10.1
* lendingclub 0.1.8
* pymongo 2.8
* psycopg2 2.5.3
* dill 0.2.2


### Installation
1. Clone this repo.
2. Download the data from the Lending Club website, or from the following 
[Dropbox address](https://www.dropbox.com/sh/pmwh81xl7bi5axv/AABSewOpldF2zdqr6JOP5lNha?dl=0), 
and place in a directory labeled `data`.
3. Install the listed requirements.
4. (Optional) Start up a MongoDB instance, and a PostgreSQL database named 'rateflask'.

To run the production version locally, run `python app.py` (or `sudo python 
app.py` should there be permission errors) in terminal from the repo directory.
For debugging, run `python app.py debug`.

To test if the installation has been successful, run `python test.py` from the same
location. To run the model against the validation set, run `python test.py 
compare`. Please note that the validation process might take some time.

### Modules

**helpers - data processing and cashflow generation**
- preprocessing.py - cleans up data and fills missing values
- postprocessing.py - creates files for charts and data table
- cashflow.py - generates cashflows and compounding curves for IRR calculation

**transfers - file input/output, API requests and database insertions**
- fileio.py - dumping and loading data with pickle/dill
- retrieve.py - requests data from Lending Club API
- database.py - inserts data to MongoDB and PostgreSQL

### Next steps

Portfolio selection model.


### License

Copyright (c) 2015 Rateflask

Licensed under the MIT licence.