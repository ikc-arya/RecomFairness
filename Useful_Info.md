# 'k' parameter

> TL;DR. K is a user-assigned parameter that defines the cutoff point (list length) while evaluating ranking quality.

- For example, you can only look at top-10 items. When evaluating the quality of recommendations or ranking, you must also define the depth of the ranked list you will look at.
- The K parameter represents the cutoff point in the list where you'd look for relevant items.
- For example, you might consider top-10 search results or top-3 recommendations since this is the app's recommendation block size.
- Or, you might look at the top 30 recommendations if you expect the user behavior to be explorative and know this to be the average number of items to view.
- This K is a use-case-specific parameter that you can set. You might also evaluate metrics at different levels of K, treating each as an additional quality measure: for example, look at top-3 and top-30 results in parallel.

# Backbone algo used

* [X] NeuMF : neural collaborative filtering method.
* [ ] GRU4Rec : sequential recommendation algorithm.
* [ ] TiSASRec : sequential and time interval aware recommendation.

# Parameters used in Final Table

1. HR@k - Hit Rate
2. NDCG@k - Normalised Discounted Cumulative Gain
3. N_Cov@k - (New) Item Coverage (time-sensitive exposure fairness)
4. Cov@k - coverage

# Steps to run the code:

Clone the original repo:

```
git clone https://github.com/Alice1998/TaFR.git
cd TaFR
```

Create the environment:

```
python -m venv tafr_env
source tafr_env/bin/activate  
# On Windows: tafr_env\Scripts\activate
```

Install dependencies

> Note: the current(latest) version of any library works (no specific requirements)

```
pip install -r requirements.txt
```
