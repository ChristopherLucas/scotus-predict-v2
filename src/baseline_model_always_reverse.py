
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


# In[3]:


# Reset output file timestamp per run
file_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Reset seed per run
numpy.random.seed(0)

# Setup training time period
min_training_years = 25
term_range = range(raw_data["term"].min() + min_training_years,
                   raw_data["term"].max()+1)

# Setup model
m = None
term_count = 0

# Iterate over all terms
for term in term_range:
    # Setup train and test periods
    train_index = (raw_data.loc[:, "term"] < term).values
    test_index = (raw_data.loc[:, "term"] == term).values
    
    if test_index.sum() == 0:
        continue
    
    # Store baseline prediction - always reverse (1)
    raw_data.loc[test_index, "baseline_predicted"] = 1


# ### Justice Accuracy - Other/Affirm/Reverse

# In[4]:


# Get index and outcomes
evaluation_index = raw_data.loc[:, "term"].isin(term_range)
target_actual = (raw_data.loc[evaluation_index, "justice_outcome_disposition"]).astype(int)
target_baseline = raw_data.loc[evaluation_index, "baseline_predicted"].astype(int)
raw_data.loc[evaluation_index, "baseline_correct"] = (target_actual == target_baseline).astype(int)

# Dummy model
print("Dummy model")
print("="*32)
print(sklearn.metrics.classification_report(target_actual, target_baseline))
print(sklearn.metrics.confusion_matrix(target_actual, target_baseline))
print(sklearn.metrics.accuracy_score(target_actual, target_baseline))
print("="*32)
print("")


# In[5]:


# Plot by term
baseline_correct_ts = raw_data.loc[evaluation_index, :].groupby("term")["baseline_correct"].mean()
baseline_correct_ts.plot()


# ### Justice Accuracy - Reverse/Not-Reverse

# In[6]:


# Get index and outcomes
evaluation_index = raw_data.loc[:, "term"].isin(term_range)
reverse_target_actual = (raw_data.loc[evaluation_index, "justice_outcome_disposition"] > 0).astype(int)
reverse_target_baseline = (raw_data.loc[evaluation_index, "baseline_predicted"] > 0).astype(int)
raw_data.loc[evaluation_index, "baseline_reverse_correct"] = (reverse_target_actual == reverse_target_baseline).astype(int)

# Dummy model
print("Dummy model - Reverse")
print("="*32)
print(sklearn.metrics.classification_report(reverse_target_actual, reverse_target_baseline))
print(sklearn.metrics.confusion_matrix(reverse_target_actual, reverse_target_baseline))
print(sklearn.metrics.accuracy_score(reverse_target_actual, reverse_target_baseline))
print("="*32)
print("")


# In[7]:


# Plot by term
baseline_reverse_correct_ts = raw_data.loc[evaluation_index, :].groupby("term")["baseline_reverse_correct"].mean()
baseline_reverse_correct_ts.plot()


# ### Case Accuracy - Reverse/Not-Reverse

# In[8]:


# Get actual and predicted case outcomes
raw_data.loc[:, "justice_outcome_disposition_reverse"] = (raw_data.loc[evaluation_index, "justice_outcome_disposition"] > 0).astype(int)
raw_data.loc[:, "baseline_predicted_reverse"] = (raw_data.loc[evaluation_index, "baseline_predicted"] > 0).astype(int)
docket_baseline_predicted_reverse = (raw_data.loc[evaluation_index, :].groupby("docketId")["baseline_predicted_reverse"].mean() > 0.5).astype(int)
docket_actual_reverse = (raw_data.loc[evaluation_index, :].groupby("docketId")["case_outcome_disposition"].mean() > 0.5).astype(int)


# Dummy model
print("Dummy model - Reverse")
print("="*32)
print(sklearn.metrics.classification_report(docket_actual_reverse, docket_baseline_predicted_reverse))
print(sklearn.metrics.confusion_matrix(docket_actual_reverse, docket_baseline_predicted_reverse))
print("Accuracy:")
print(sklearn.metrics.accuracy_score(docket_actual_reverse, docket_baseline_predicted_reverse))
print("="*32)
print("")


# In[9]:


# Create merged docket dataframe
docket_df = pandas.concat([docket_actual_reverse, docket_baseline_predicted_reverse], axis=1)
docket_df.columns = ["outcome_actual", "outcome_predicted"]
docket_df.loc[:, "baseline_reverse_correct"] = (docket_df["outcome_actual"] == docket_df["outcome_predicted"]).astype(int)
docket_df = docket_df.join(raw_data.loc[evaluation_index, ["docketId", "term"]].groupby("docketId")["term"].mean())
docket_df.head()


# In[10]:


# Plot by term
baseline_case_reverse_correct_ts = docket_df.groupby("term")["baseline_reverse_correct"].mean()
baseline_case_reverse_correct_ts.plot()


# In[11]:


# Output model results
raw_data.to_csv("../data/output/raw_docket_justice_baseline_model_always_reverse.csv.gz", compression="gzip")

