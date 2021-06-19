### --------------------------------------------------------------------------------------------------------
# Feed-Forward Neural Network - Loan Payback
### --------------------------------------------------------------------------------------------------------
### Feed-Forward Neural Network:
- Abbreviation: **FNN**
- Expand of the single Perceptron Model (which is build out based on a Biological Neuron Model) and create a Multi-layer Perceptron Model (also known as basic Artificial Neural Network (ANN))
- **Structure of a Feed-Forward Neural Network** (basic ANN):		
	- **Input layer** - first layer that directly accepts real data values (inputs of the raw data)
    - **Hidden layers** - layers between Input and Output layers (Black Box)
	- **Output layer**	- final estimate of the output (closely associated with the label that we are trying to predict)
- **Create a Feed-Forward Neural Network** (basic ANN):
    - take the outputs of the vertical layer of single Perceptron (node)
    - feed the outputs into as inputs to the next layer of Perceptrons (nodes) -> outputs of the previous layer become the inputs of the next layer
- **Information moves in one direction (forward)** and **connections between the nodes do not form a cycle or loops**
- Neural Network becomes “**Deep Neural Network**” if **it contains 2 or more Hidden layers**:				
	- **width of the Neural Network** - how many neurons (Perceptrons) are in the layer
	- **depth of the Neural Network** - how many layers are in total
- How **Neural Network works**:
    - Take in inputs x
    - Multiply the inputs x by weights w, and add biases b
    - Pass the result through an Activation Function
    - Estimated output ŷ at the end of all layers -> model’s estimation of what it predicts the label to be
    - Evaluate the prediction (estimated output ŷ) against true value of the label y -> Cost Functions and Gradient Descent
    - Update the Neural Network’s weights w and biases b -> Backpropagation

### --------------------------------------------------------------------------------------------------------
### Project Objective: Prediction of loan payback
Create and train a Feed-Forward Neural Network model for Deep Learning classification that allows to put in a set of data and returns back a prediction (classification) whether or not someone will pay loan back to investor based on historical information. This way, in the future, when we get new potential customers, we can assess whether or not they are likely to pay back the loan.

Information about the loan payback data is in the subset of the LendingClub dataset 'Lending_Club_Loan_two.csv' and is obtained from [Kaggle.com](https://www.kaggle.com/wordsforthewise/lending-club)

Lending Club is a peer-to-peer lending company, headquartered in San Francisco, California, that matches borrowers with investors through an online platform. It services people that need personal loans between 1,000 US dollars and 40,000 US dollars. Borrowers receive the full amount of the issued loan minus the origination fee, which is paid to the company. Investors purchase notes backed by the personal loans and pay Lending Club a service fee. The company shares data about all loans issued through its platform during certain time periods. It is the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.

The example demonstrates how to:
- Load and explore data
- Define the network architecture
- Specify training options
- Train the network
- Predict the labels of new data and calculate the classification accuracy

**Data Overview**

----
-----
There are many LendingClub data sets on Kaggle. Here is the information on this particular data set:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LoanStatNew</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>loan_amnt</td>
      <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>term</td>
      <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>int_rate</td>
      <td>Interest Rate on the loan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>installment</td>
      <td>The monthly payment owed by the borrower if the loan originates.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>grade</td>
      <td>LC assigned loan grade</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sub_grade</td>
      <td>LC assigned loan subgrade</td>
    </tr>
    <tr>
      <th>6</th>
      <td>emp_title</td>
      <td>The job title supplied by the Borrower when applying for the loan.*</td>
    </tr>
    <tr>
      <th>7</th>
      <td>emp_length</td>
      <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>home_ownership</td>
      <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
    </tr>
    <tr>
      <th>9</th>
      <td>annual_inc</td>
      <td>The self-reported annual income provided by the borrower during registration.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>verification_status</td>
      <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
    </tr>
    <tr>
      <th>11</th>
      <td>issue_d</td>
      <td>The month which the loan was funded</td>
    </tr>
    <tr>
      <th>12</th>
      <td>loan_status</td>
      <td>Current status of the loan</td>
    </tr>
    <tr>
      <th>13</th>
      <td>purpose</td>
      <td>A category provided by the borrower for the loan request.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>title</td>
      <td>The loan title provided by the borrower</td>
    </tr>
    <tr>
      <th>15</th>
      <td>zip_code</td>
      <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
    </tr>
    <tr>
      <th>16</th>
      <td>addr_state</td>
      <td>The state provided by the borrower in the loan application</td>
    </tr>
    <tr>
      <th>17</th>
      <td>dti</td>
      <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
    </tr>
    <tr>
      <th>18</th>
      <td>earliest_cr_line</td>
      <td>The month the borrower's earliest reported credit line was opened</td>
    </tr>
    <tr>
      <th>19</th>
      <td>open_acc</td>
      <td>The number of open credit lines in the borrower's credit file.</td>
    </tr>
    <tr>
      <th>20</th>
      <td>pub_rec</td>
      <td>Number of derogatory public records</td>
    </tr>
    <tr>
      <th>21</th>
      <td>revol_bal</td>
      <td>Total credit revolving balance</td>
    </tr>
    <tr>
      <th>22</th>
      <td>revol_util</td>
      <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
    </tr>
    <tr>
      <th>23</th>
      <td>total_acc</td>
      <td>The total number of credit lines currently in the borrower's credit file</td>
    </tr>
    <tr>
      <th>24</th>
      <td>initial_list_status</td>
      <td>The initial listing status of the loan. Possible values are – W, F</td>
    </tr>
    <tr>
      <th>25</th>
      <td>application_type</td>
      <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
    </tr>
    <tr>
      <th>26</th>
      <td>mort_acc</td>
      <td>Number of mortgage accounts.</td>
    </tr>
    <tr>
      <th>27</th>
      <td>pub_rec_bankruptcies</td>
      <td>Number of public record bankruptcies</td>
    </tr>
  </tbody>
</table>

---
----

### --------------------------------------------------------------------------------------------------------
### Table of Contents:
1. File Descriptions
2. Technologies Used
3. Structure of Notebook
4. Executive Summary

#### 1. File Descriptions
- FeedForward Neural Network - Loan_Payback.ipynb
- Lending_Club_Loan_two.csv
- Lending_Club_Loan_info.csv
- README.md

#### 2. Technologies Used
- Python
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-Learn

#### 3. Structure of Notebook
1. Import the Libraries
2. Load the Data
3. Exploratory Data Analysis
    - 3.1 Check out the Data
    - 3.2 Data Visualization
4. Data Preprocessing and Feature Engineering
    - 4.1 Identify the variables
    - 4.2 Dealing with Missing values
    - 4.3 Dealing with the Categorical features
    - 4.4 Dealing with the Non-numerical features
5. Model Building
    - 5.1 Split of columns
    - 5.2 Split the data into Training dataset and Testing dataset
    - 5.3 Normalize the values of variables in the input Training dataset
6. Feed-Forward Neural Network architecture
    - 6.1 Create and Define the Feed-Forward Neural Network model
    - 6.2 Create a Callbacks
      - 6.2.1 Create a Callback for Early Stopping
      - 6.2.2 Create a Callback to Save the best model from validation
      - 6.2.3 Create a list of Callbacks
    - 6.3 Train / fit the model
    - 6.4 Load the best model from the validation
    - 6.5 Evaluate the model on Training data
      - 6.5.1 Visualization of Model Accuracy
      - 6.5.2 Visualization of Model Loss
    - 6.6 Predictions from the model on Testing data
    - 6.7 Evaluate the model on Testing data
      - 6.7.1 Calculate the Accuracy, Cross-entropy Loss and FNN Error
      - 6.7.2 Classification report
      - 6.7.3 Confusion matrix
      - 6.7.4 Interpreting of Coefficient of the features
    - 6.8 Prediction on New Data

#### 4. Executive Summary
TBA
