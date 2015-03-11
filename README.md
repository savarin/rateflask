
**Helpers:**

preprocessing.py
- Cleans up data and fills in missing values

cashflow.py
- Generates expected cashflows and interest rate compounding curves, for IRR calculation


**Models:**

currentmodel.py
- Main model
- Input is data of 3-year loans issued between 2012 and 2014, i.e. not yet matured
- Calculates expected IRR based on features (e.g. FICO score)

maturedmodel.py
- Model to generate validation data
- Input is data of 3-year loans issued between 2009 and 2011, i.e. already matured
- Calculates actual IRR based on payments made, i.e. feature independent
- Actual IRR used to validate currentmodel.py

rankingmodel.py
- Initial idea of ranking loans within sub-loan grade band, simple workable model

**API-related:**

retrieve.py
- Retrieves data from API, and processes in format that preprocessing.py can accept

app.py
- Actual app, makes request using helper functions in retrieve.py and uses models trained by currentmodel.py to calculate expected IRR.


**Analysis:**

modeltest.py
- Tests currentmodel.py and maturedmodel.py are working

modelfit.py
- Trains model in currentmodel.py, then dumped to pickle for use in calculating expected IRR in modelpredict.py and app.py

modelpredict.py
- Calculates actual IRR (as per maturedmodel.py) and expected IRR (as per currentmodel.py)