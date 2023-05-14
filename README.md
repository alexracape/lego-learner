# lego-learner
Using machine learning and principles from finance to trade lego sets

## The Data
This project uses a combination of data from multiple sources
- Features and list price from 1970 - 2015 from [seankross](https://github.com/seankross/lego) on github
- Current Price - Bricklink API as of 05/10/2023
- Recent list prices and features - Brickset API as of 05/10/2023

## The Model
TBD - Exploring random forest and neural network

## The Results
TBD

## Ideas to Explore

- Use price of all pieces as 'book value' indicator
- Construct benchmark to compare results with
- Can we spin as fundamental analysis?
- Need more finance content...
- Could we get number of sets out there and construct market cap?

## Plan Going Forward
- Figure out best hyperparameters and tune models to predict set value
- See if we can find number of sets manufactured
- Use model to identify undervalued sets
  - Ex: model predicts price should be more than it is listed as
  - Can frame as fundamental analysis
- Buy undervalued sets and see how the cohort does, maybe go by year?
  - Ex: train on all except 1995, then buy sets from that year
- Compare returns to benchmark and S&P 500 for kicks
- Can also experiment with forecasting into the future with timedelta as a parameter
