# Mid-term-Project

## Project/Goals

The overall goal of this project is for us to utilize all of the skills we have learned so far in the Lighthouse Labs bootcamp. The project was open ended in the sense that we (Susmitha and Samuel) needed to find a dataset of interest, perform the data cleaning process, carry out EDA and finally make some form of predictive analysis.

This repo contains three folders, data cleaning, data viz and model. Each of the folders contains notebooks for importing the dataset, cleaning and exporting it to a CSV file after cleaning. The second folder contains a Tableau file were data visualizations and EDA process were done. The third folder contains the ML model notebook, python script for embedding the preprocessed model dataset and some images used in the Tableau visualizations.

We selected the Los Angeles crime dataset which we downloaded from the US government open data website [Crime data](https://catalog.data.gov/dataset/crime-data-from-2020-to-present). The dataset contains reported cases of assault from 2020 up to the present time we downloaded the data (October 2nd 2023).

## Process

### Data was downloaded in CSV format. There were 28 attributes, mostly categorical variables and only one continuous variable (age).
1. The dataset has the shape (807377, 28) before the data cleaning process. After the cleaning process, the shape reduced to (589265, 19). The dataset with this dimension was saved to a CSV file and imported into the model_building.ipynb notebook. Further preprocessing for the model reduced the dimensions to (43117, 6). The dataset was too large for the model, so we had to predict based on 2023 data alone. The CSV file with dimensions (589265, 19) was imported to Tableau for EDA and data visualization insights.

2. The following features were selected for the predictive model:
 - `[Area name]`
 - `[Crime description]`
 - `[Victim's sex]`
 - `[Victim's age]`
 - `[Victim's race]`
 - `[Crime code 1]`


### Explored the data, cleaned it from inappropriate values such as victim age < 0, latitude and longitude having values of 0.0. We reduced the sex to study only male and female. Dropped columns that have no connection to our questions of interest. Renamed columns to human comprehensible names. Carried out initial exploration of the data. In general, we performed a bunch of data preprocessing.

1. The first preprocessing began with the exploratory_analysis_clean.ipynb notebook, where we did applied standard approaches of data analysis. The final DataFrame was saved to a CSV for both the machine learning model and data visualization in Tableau.

2. The second preprocessing was done while analyzing the dataset to ensure it meets the selected model requirements, dealing with outliers in the `[Victim's age]`, so the machine learning model can give more precise predictions. We also carried out a bunch of standard data preparation approaches such as boxplot for outliers, histogram for numerical features and multivariate analysis for categorical features. Further filtering was done to trim down the dataset to a number that the machine learning model can process without raising a MemoryError.
 - Many codes in the model_building.ipynb notebook was commented because the output is really long. Please uncomment them and run the cell to see the outputs.


### Performed exploratory data analysis (EDA) process, visualized the data to make inference regarding possible trends and/or patterns (both usual and unusual trends). The questions we were seeking to answer are as follows:
1. Is the distribution of crime biased towards a particular gender?
2. Does the crime severity have an impact on the number of crimes committed or most of the crime are petty crimes?
3. What does the trend of crime look like over the three years and what are the top crimes?
4. Is a particular race a target or is the reported assault cases evenly dsitributed across race?
5. What kind of assault weapons are used against the range of average age?
6. Are there particular days that crime rates are higher?
7. What is the precentage of the each assault case investigation status relative to the total reported assault cases?


### We could not use the Liner Regression or Logistic Regression models for our predictive analysis because we only had one numerical (continuous) feature, all other features of interest were multi-class categorical variables. This pushed us to search for models outside of what we've been exposed to. Came across the LLM + KMeans model and used it to perform predictive policing analytics.
 - After data preparation and further preprocessing, we saved the model ready data as a CSV file, imported it into the embedding.py python script and used the `SentenceTransformer` to embed the features into vectorized high dimensional data. Prepared high dimensional data was exported as a CSV file.
 - The embedded model data CSV was imported into a DataFrame in the model notebook. The `ECOD` method (empirical cumulative distribution functions for outlier detection) from the Python Outlier Detection (`PyOD`) library was used to detect and remove outliers.
 - The KMeans model was instatiated and an optimal number of clusters was determined.
 - The embedded data was fitted to the model and the clusters were generated.


### Provided valuable insights from EDA process using Tableau and made predictions using model built in python.
#### The types of visualizations included in the project are:
 - Bar charts.
 - Histograms.
 - Trend analysis through time.
 - Heat map.

Some of the categorical variables in the dataset were not defined. For example the definitions of the `[Victim's race]` are as follows:

A - Other Asian, B - Black, C - Chinese, D - Cambodian, F - Filipino, G - Guamanian, H - Hispanic/Latin/Mexican, I - American Indian/Alaskan Native, J - Japanese, K - Korean, L - Laotian, O - Other, P - Pacific Islander, S - Samoan, U - Hawaiian, V - Vietnamese, W - White, X - Unknown, Z - Asian Indian.

We couldn't find the description for the `[Crime code 1]` column anywhere online (it's probably classified information). What we could gather from the website describing each variable is that the lower the number the more severe the assault committed against a victim. Basically, more like a severity index.

## Results
The results of the data visualization were presented using a story. The story begins from the distribution of the age of assault victims classified by sex. Then it shows insights into which race is mostly a victim in assault crimes. It proceeds to explain the severity of crimes across areas in Los Angeles and the kind of weapons used against victims. It also investigates the distirbution of crimes across days of the week to find what day has the highest number of crimes. The story ends with predictions using cluster related crime information and recommendation.

### Key takeaways
 - Assault crimes are almost evenly distributed across ages for both males and females.
 - The Hispanics usually fall vcitim of assault compared to all other cases. Second is the Whites, then the Blacks, Other (undefined) and Other Asians. These are the top five assault victim's races.
 - The total number of severe crimes are not a huge proportion of the total crimes. The most common crimes are identity theft, battery - simple assault.
 - Crime rates are higher in the Central and 77th street.
 - The ML model created five clusters and three of those clusters identified Central area as a crime hotspot. Two of the clusters identified gender-based violence as the top crimes, three of the clusters identified females as the most targeted victims. All five clusters identified the age of victims to be between 34 and 37. Three of the clusters identified that the top crimes to be committed are not very severe. Probably petty crimes.


## Challenges 
 - Finding the right model for the predictive analysis we had in mind.
 - Failure to obtain insights using the maps despite having longitude and latitude coordinates in the dataset. Tableau was misinterpreting the coordinates as a dimension and not a measure. When we were finally able to get that right, we couldn't get valuable information from the maps created due to not having country and state defined in the dataset. Unfortunately, time was a constraint, so we couldn't add those columns.
 - Not enough time to clean the data more thoroughly. We discovered some discrepanices while we were performing EDA and data visualization.

## Future Goals
We used only the top five crimes across Los Angeles in the `[Crime description]` for the ML model. Expanding this number, introducing other features into the model are future goals.


Please reach out to either of the contributors if you have concerns, questions or recommendations regarding this project. The following article by [Damian Gill](https://webcache.googleusercontent.com/search?q=cache:https://medium.com/towards-data-science/mastering-customer-segmentation-with-llm-3d9008235f41) was instrumental to the success of the ML model presented in this project.
