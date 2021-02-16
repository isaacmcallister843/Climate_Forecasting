# Climate_Forecasting


## Motivation 
Climate predictions can be a critical tool in environmental modeling and habitat modeling. This project uses LSTMs to predict temperature, humidity, and precipitation for a mine site. The technology here is a proof of concept, and the end goal is to apply multivariate LSTMs to a variety of other important climate variables. This project was a great learning experience, however not particular complicated. 

## Data and Modeling
The climate station is from a met station installed on the mine site by a contractor. The first issue is that LSTMs needs thousands of data points to train on, while the field data is limited to several hundred. My solution was to train the model on publicly available climate data from the government of Canada, and then apply the model to our dataset. This worked pretty not great as can be seen in the figure below, but it's a starting point. My next solution was to deploy a random forest algorithm to try and close the difference. This model took the LSTM predictions as independent variables and the actual values as dependent variables. The "Random Forest Correction" line is the predictions made from this method. This works pretty well, however we hit a wall when trying to forecast far into the future. The random forest algorithm is trained on too small of a sample size and can't correct accurately. 

The main issue with the approach I took was I chose a data heavy method (LSTM + Random Forest) for a project that didn't have enough data. If I would do this project again I would probably just use Prophet, which expects seasonal trends and is much more lightweight. Prophet doesn't support multivariable forecasts however. I hope to revisit this project when 2020 and 2021 field data is available, the model should perform a lot better. 

![Results](https://user-images.githubusercontent.com/78721353/108029417-41132d80-6fe2-11eb-9b09-45f566d5328b.png)
#### Figure 1 
