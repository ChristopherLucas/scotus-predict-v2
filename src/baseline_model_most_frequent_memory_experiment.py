
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')

# Imports
import matplotlib.pyplot as plt
import pandas
import seaborn
seaborn.set_style("darkgrid")

# Project imports
from legacy_model import *


# In[2]:


# Get raw data
raw_data = get_raw_scdb_data("../data/input/SCDB_Legacy_01_justiceCentered_Citation.csv")


# In[16]:


# Reset output file timestamp per run
file_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Reset seed per run
numpy.random.seed(0)

# Setup training time period
min_training_years = 25
term_range = range(raw_data["term"].min() + min_training_years,
                   raw_data["term"].max()+1)

baseline_window_range = range(2, 50)
for window in baseline_window_range:
    raw_data.loc[:, "baseline_{0}_predicted".format(window)] = numpy.nan

# Iterate over all terms
for term in term_range:
    print(term)
    
    # Outer loop constants and tests
    test_index = (raw_data.loc[:, "term"] == term).values
    if test_index.sum() == 0:
            continue

    # Iterate over window sizes
    for baseline_window in baseline_window_range:
        # Baseline training index
        baseline_train_index = ((raw_data.loc[:, "term"] < term) & (raw_data.loc[:, "term"] >= (term-baseline_window))).values
        
        # Fit the "baseline" model
        d = sklearn.dummy.DummyClassifier(strategy="most_frequent")
        d.fit(numpy.zeros_like(raw_data.loc[baseline_train_index, :]), 
              (raw_data.loc[baseline_train_index, "justice_outcome_disposition"]).astype(int))

        # Store baseline predictions
        raw_data.loc[test_index, "baseline_{0}_predicted".format(baseline_window)] = d.predict(numpy.zeros_like(raw_data.loc[test_index, :]))


# ### Justice Accuracy - Reverse/Not-Reverse

# In[17]:


# Store data across all windows
justice_window_accuracy_data = []

# Iterate over all windows
for window in baseline_window_range:
    # Get index and outcomes
    evaluation_index = raw_data.loc[:, "term"].isin(term_range)
    reverse_target_actual = (raw_data.loc[evaluation_index, "justice_outcome_disposition"] > 0).astype(int)
    reverse_target_baseline = (raw_data.loc[evaluation_index, "baseline_{0}_predicted".format(window)] > 0).astype(int)
    
    # Append to dataset
    justice_window_accuracy_data.append((window, sklearn.metrics.accuracy_score(reverse_target_actual, reverse_target_baseline)))
    
# Create dataframe
justice_window_accuracy_df = pandas.DataFrame(justice_window_accuracy_data, columns=["M", "accuracy"])
justice_window_accuracy_df.set_index(justice_window_accuracy_df["M"], inplace=True)
del justice_window_accuracy_df["M"]
justice_window_accuracy_df.plot()


# ### Case Accuracy - Reverse/Not-Reverse

# In[20]:


# Store data across all windows
case_window_accuracy_data = []

# Outer loop
raw_data.loc[:, "justice_outcome_disposition_reverse"] = (raw_data.loc[evaluation_index, "justice_outcome_disposition"] > 0).astype(int)
docket_actual_reverse = (raw_data.loc[evaluation_index, :].groupby("docketId")["case_outcome_disposition"].mean() > 0.5).astype(int)

    
# Iterate over all windows
for window in baseline_window_range:
    # Get index and outcomes
    raw_data.loc[:, "baseline_predicted_reverse"] = (raw_data.loc[evaluation_index, "baseline_{0}_predicted".format(window)] > 0).astype(int)
    docket_baseline_predicted_reverse = (raw_data.loc[evaluation_index, :].groupby("docketId")["baseline_predicted_reverse"].mean() > 0.5).astype(int)

    
    # Append to dataset
    case_window_accuracy_data.append((window, sklearn.metrics.accuracy_score(docket_actual_reverse, docket_baseline_predicted_reverse)))
    
# Create dataframe
case_window_accuracy_df = pandas.DataFrame(case_window_accuracy_data, columns=["M", "accuracy"])
case_window_accuracy_df.set_index(case_window_accuracy_df["M"], inplace=True)
del case_window_accuracy_df["M"]
case_window_accuracy_df.plot()


# In[22]:


# Output model results
raw_data.to_csv("../data/output/raw_docket_justice_baseline_model_most_frequent_memory_experiment.csv.gz", compression="gzip")

