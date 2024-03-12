# ProjectTitle: COGNIZANT-ARTIFICIAL-INTELLIGENCE-VIRTUAL-INTERNSHIP-Machine-Learning-for-Gala-Grocery-Retail

Dataset link:https://www.theforage.com/virtual-experience/5N2ygyhzMWjKQmgCK/cognizant/artificial-intelligence-rtbq/exploratory-data-analysis

Dashboard:
[Download Video]([https://raw.githubusercontent.com/YourUsername/YourRepository/main/video.mp4](https://github.com/itsmesethus/COGNIZANT-ARTIFICIAL-INTELLIGENCE-VIRTUAL-INTERNSHIP-Machine-Learning-for-Gala-Grocery-Retail-/blob/main/dashboard/dashboard%20screen%20capture.webm))


*************************************************************************************************************************************

## OBJECTIVE:
  The objectives of the Cognizant Virtual Experience Program:

      1. Gain practical experience in data exploration, analysis, and modeling using Python and develop skills in framing business problems, creating strategic plans, and communicating findings.

      2. Build predictive models by combining, transforming, and modeling multiple datasets to address specific business requirements.

      3. Develop machine learning algorithms for predictions, and evaluating and improving model performance.

## VARIABLES:

**timestamp**: The date and time when a particular transaction or event occurred.

**prd_id**: The unique identifiers or codes for different products.

**category**: The category or type of product.

**customer_type**: This column represents the type of customer, such as individual, business, or any other categorization of customers.

**unit_price**: The price of a single unit of the product.

**quantity**: The number of units of the product purchased or sold in a particular transaction.

**total**: The total amount or revenue generated from a particular transaction, calculated by multiplying the unit_price by the quantity.

**payment_type**: This contains the mode of  transaction, such as cash, credit card, or any other payment method.

**avg_stk_prc**: The average stock price or cost price of the product.

**temperature**:  The temperature at the time of the transaction or event.


## LIBRARIES USED:

   - Pandas, Matplotlib, Seaborn, Scikit-learn, Scipy, datetime,Tensorflow, Numpy, Keras.


## TABLE OF CONTENTS:

- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Feature Engineering](#Feature-Engineering)
- [Model Building](#Model-Building)
- [Model Evaluation Metircs](#Model-Evaluation-Metircs)
- [Insights](#insights)
      
## Introduction:

  Gala Groceries is a renowned supermarket chain that operates stores across multiple regions. As part of its efforts to enhance customer experience and optimize business operations, the company has partnered with Cognizant, a leading professional services firm, to leverage data analytics and machine learning techniques. Gala Groceries' broader goal of enhancing customer satisfaction and optimizing business operations through data-driven insights.

## Data Preprocessing:

   In this part, I have inquired about the data quality such as missing values, inconsistent column, infered the proper format for datetime and inquired about the duplicated rows in the dataset. Also did descriptive statistics for both categorical and numerical features.

## Data Visualization:
   
   For Data visualization part,

   * Barplots for Categories, cutomer type, Payment types, top purchasing timings and respective customers behaviours.

   * Boxplots for all  numerical features to asses the outliers or anomalies in the dataset.

   * Histograms and KDE plots for the numerical features to asses the spread or the distributions of the dataset. 

   * Regplot for  unit_price, quantity and total

   * Aso did Correlation matrix for the above three features to know about its relationships.
 
## Statistical Test- Chisquare Test(test for independence):

   * To determine whether there is a significant association between two categorical variables.
   *  It assesses whether the observed frequencies of the categories in one variable are independent of the categories in the other variable, or if there is a relationship between them.
    Hypothesis St,

    - Null Hypothesis (H0): There is no association between the two categorical variables.
    - Alternative Hypothesis (H1): There is an association between the two categorical variables.
   
   * Calculated the E(expected fequency) for each cell in a contingency table using Scipy.stats library.
   * Chisq = (O - E)^2/E was the test statistic.
   * Decision: If p val > 0.05 ---> Falied to reject the H0 based on the samples.
               If p val <= 0.05 ----> Falied to accept the H0 based on the samples
   * Effect Size:

         * 0 --> Weak Association
         * 0.1-0.3 --> Small to Moderate Association
         * 0.3-0.5 --> Moderate to Strong Association
         * close to 1 --> Strong Association


# Feature Engineering:

   Before entering into Feature Engineering part, data must be cleaned from the outliers. For outliers cleansing, I have used the IQR(Inter-Quartile Range) method.
    
    IQR = (Q3-Q1)
    Lower Bound = Q1 - 1.5 * IQR
    Upper Bound = Q3 + 1.5 * IQR

   Using the mask = (data >= Lower_bound) & (data <= Upper_bound) data will be cleaned from the anaomolies.

* Though the distributions of the data was not quite normal, i tried the Normalizer from the sklearn library which offeres l1 normalization using this i converted most of the features to closer to normal distributions. Additionally inquired with the QQ-plot for Normality test using data visualization.

* Converted the categorical features to numerical coding using LabelEncoders and other few categorical variables for One-Hot Encoding techniques.

* For Independent features tried to know about the which are the features are more import using Mututal Information technique for Regression and Anova methods for features.

## Model Building

* For Train-test split done. Size of test is 20% percent of data with shuffled ones.

* To build a Ml model with well optimal model to generalize the unseen data Cross validation with hyperparameter tuning technique called RandomizedSearch method.

* ML MODEL FOR RECOMENDATION BASED ON ITS PERFORMANCES

   - Adaptive Boosting Regressor
   - Hist Gradient Boost Regressor
   - Bagging Regressor
   - Radius Neighbors Regressor
   - Artificial Neural Network - Multiple Linear Regression Model

## Model Evaluation Metircs

   * R Square
   * Mean Squared Error
   * Mean Absolute Error
   * Median Absolute Error
   * Rooted Mean Square Error.

## Insights

* Fruits, Vegetables, Packaged foods and baked goods were the top selling categories
* Non-Members of customer type were the people who highly visiting the store.
* Unit price vs Quantity (0.0246)-which is a weak positive correlation between them.
Quantity vs Total (0.521)- which says that there is a Modereate positive correlation btw them.
Total vs Unit price (.792)- which implies very strong positive relationship between these two features.
* Larger differences in mean quantity but Cheese,bevarages and seafoods solded higher
* In unit price, same categories such as medicine, seafood and kitchen are the higher with mean total sales.
* Peak sale timing happening in the 14th hour and 16th hour of day.
* Based on the peak sale timing , customer types:
      
      * Premium customers visiting mostly - 18th and 12th hour of the day
      * Standard customers visiting mostly - 16th hour of the day
      * Both Basic and Non-members visiting mostly - 11th hour of the day
* Mode of Payments across hours

      * Cash mode is highly used on -19th and 16th hour of the day.
      * Debit card mode is highly pursued on -18th hour of the day(premium customers)
      * Credit card mode is highly used on -9th and 13th hour of the day
      * E-wallet mode is highly used on -12th hour of the day

* In almost every customer type category the highest number of quantity consumed by all the type is '4, people prefering more in quanity and the order (4 > 3 > 2> 1)
* For Chi2 test of independence:
     Since all the p value(pearson) are lower than 0.05 so there is a relationship between Payment Mode  and Quantity. When considering the effect size , it has very weak association between the payment type and quantity variables.
