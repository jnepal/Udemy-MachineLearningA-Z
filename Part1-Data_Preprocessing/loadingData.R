# Data Preprocesing

# Importing the datasets
dataset = read.csv('Part1-Data_Preprocessing/data/Data.csv')

# Taking Care of missing data

# Column Age

# If else parameter
# firstparam : condition to be checked
# secondparam : statment if condition is TRUE
# thirdparam: statement if condition is FALSE

# is.na() checks if the value is missing
# na.rm = TRUE to include the empty column into consideration while calculating mean
# mean is builtin function

dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age,FUN = function (x) mean(x, na.rm = TRUE)),
                     dataset$Age)

# Column Salary
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function (x) mean(x, na.rm = TRUE)),
                        dataset$Salary)

# Encoding Categorical data
# No need for dummy encoding in R

dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))

dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))

# Splitting data into training and testing set
# install.packages('caTools')

library(caTools)
set.seed(123) # same as random_state in train_test_split
split = sample.split(dataset$Purchased, SplitRatio = 0.7) # training data = 70% of total data. return TRUE if the data is assigned to training set and FALSE if it is assigned to test

trainingSet = subset(dataset, split == TRUE)
testingSet = subset(dataset, split == FALSE)

# Feature Scaling

# Country and Purchased Column is factored. A factor is R is not a number
# Inorder to scale, we need the feature must be integer
# So we will scale columns other than Country and Purchased
trainingSet[, 2:3] = scale(trainingSet[, 2:3])
testingSet[, 2:3] = scale(testingSet[, 2:3])