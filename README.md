# Machine Learning Prediction of Line Loading

![Predictions](https://github.com/Fernando3161/pandapower_ml/blob/main/data/ScatterPlot.png)

## Introduction

Welcome to the `pandapower_ml` repository! This project aims to perform machine learning on a power grid to predict the loading of power lines in a district electric grid using meteorological information. By leveraging the capabilities of the `pandapower` and `sklearn` frameworks, we can make accurate predictions that help manage the power grid more efficiently.

## Approach

The approach of this project involves using machine learning algorithms to analyze the relationship between power line loading and meteorological data. By training the models on historical data, we can create predictive models that estimate the power line loading based on current and forecasted weather conditions. This information can be valuable for grid operators to plan and optimize the power distribution process effectively.

## Dependencies

To run this project, you need to have the following frameworks installed. Please make sure that the requirements from `"requirements.txt"` are installed by running the following command:

```
pip install -r requirements.txt
```


The required dependencies include:

1. `"pandapower"`: A Python library for power system modeling and analysis.
2. `"sklearn"`: A machine learning library in Python.

## Installation and Execution

To install and execute the `"pandapower_ml"` project, follow these steps:

1. Clone this repository to your local machine using the following command:

```
git clone https://github.com/Fernando3161/pandapower_ml.git
```

2. Navigate to the project directory:

```
cd pandapower_ml
```

3. Install the required dependencies using `pip`:

```
pip install -r requirements.txt
```

4. Run the `"main.py"` script to perform the power line loading prediction:

```
python main.py
```

## Results

The analysis of the project indicates that the lowest errors in power line loading predictions are obtained when considering between 10 and 11 features of the data and using a training size of 80%.
<img src="https://github.com/Fernando3161/pandapower_ml/blob/main/data/heatmap.png"  width="60%" height="60%">

The values of test and prediction for several lines can be then compared:
![Test and Predicitons](https://github.com/Fernando3161/pandapower_ml/blob/main/data/Test-v-Prediction.png)

## Acknowledgments

We would like to express our gratitude to all the contributors and researchers whose work has been instrumental in the development of this project. We also extend our thanks to the open-source community for providing valuable resources that have been essential in building this repository.

## Author

This project was created and developed by Fernando PV.

For any questions or feedback, feel free to reach out to the author via email: fernandopenaherrera@gmail.com.




