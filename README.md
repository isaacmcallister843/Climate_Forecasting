# Climate_Forecasting


## Motivation 
Climate predictions can be a critical tool in enviromental modeling and habitat modeling. This project uses LSTMs to predict temperature for a mine site. The technology here is a proof of concept, and the end goal is to apply multivariant LSTMs to a variety of important climate variables. This project was a great learning experience, however not particular complicated. 

## Data and Modeling
The actual climate data is the same dataset as the heatmapping project. The issue is that LSTMs needs thousands of data points to train on, while the feild data is limited to several hundred. My solution was to train the model on publically avalible climate data from the goverment of Canada, and then apply the model to our dataset. This worked pretty not great as can be seen in the below figure, but its a starting point. My next solution was the deploy a randomforest algorithem to try and close the difference. This model took the LSTM predictions as independent variables and the actaul values as dependent variables. The "Random Forest Correction" line is the predictions made off this method. This works pretty well, however we hit a will when trying to forcast far into the future. The randomforest algorithem is trained on too small of a sample size and can't forcast accurately into the future. 

The main issue with the approach I took was I chose a data heavy method (LSTM + Random Forest) for a project that didn't have enough data. If I would do this project again I would probably just use Prophet, which expects seasonal trends and is much more lightweight. Prophet doesn't support multivariable forecasts however. I hope to revist this project when 2020 and 2021 feild data is avalible, the model should perform alot better. 

![Results](https://user-images.githubusercontent.com/78721353/108029417-41132d80-6fe2-11eb-9b09-45f566d5328b.png)
#### Figure 1 
