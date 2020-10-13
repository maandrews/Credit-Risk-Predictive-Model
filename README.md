# Credit-Risk-Predictive-Model
Credit risk predictive model using stacking.  AUROC is used to evaluate performance.

This is based on the Kaggle competition "Give Me Some Credit".  The training and test data can be found at 
https://www.kaggle.com/c/GiveMeSomeCredit/data

The model I uploaded uses a stacking approach where the first layer contains multiple different classifiers, including a multilayer perceptron and trees.  The final predictions are done by a layer of XGBoost.  Features from the data were a bit thin initially so it was necessary to add a good amount more to increase prediction accuracy.  

The model gives an AUROC of 0.86711.  The winning team in the competition had an AUROC of 0.86955.  I ended up not doing too
bad for a weekend project, top 12% score among nearly 1000 teams.
