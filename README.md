# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project is about predicting customer churn for a bank. 

It has two scripts, namely churn_library.py and churn_script_logging_and_tests.py. 

The churn_library.py contains all functions. The churn_script_logging_and_tests.py contains the tests and logging for the churn_library.py. 

They can be run as standalone scripts. The churn_library.py has a similar structure as the accompanied Jupyter Notebook. 

The code has been formatted using autopep8 and pylint.

## Running Files

To run the files a few steps have to be taken:

* Create the virtual env
```bash
python3 -m venv venv
```

* Activate the virtual env

   - Windows: `my_venv\Scripts\activate.bat`
   - Linux/macOS: `source my_venv/bin/activate`

Poetry has been used for dependency management. 

### Install dependencies

* Run the poetry.lock file

```bash
poetry install
```

* Run the churn_library.py script

```bash
python churn_library.py
```

* Run the churn_script_logging_and_tests.py script

```bash
python churn_script_logging_and_tests.py
```