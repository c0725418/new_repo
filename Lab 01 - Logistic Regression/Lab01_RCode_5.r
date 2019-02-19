library("caTools")

#---------------------------------
#1. Reading and printing data in CSV file.
#---------------------------------
# Load the dataset from: 

dataset_voice= read.csv("D:/Lambton College/Term 3/Introduction to Artificial Intelligence/Labs/Lab 01/voice.csv")

View(dataset_voice)
dim(dataset_voice)

#---------------------------------
#2. Data munging
#---------------------------------

dataset_voice$label <- ifelse(dataset_voice$label=="male",1,0)
#dataset_voice$label <- as.factor(dataset_voice$label)

#---------------------------------
#3 Split the data
#---------------------------------

split <- sample.split(dataset_voice, SplitRatio = 0.8)
split
s_train <- subset(dataset_voice, split = "TRUE")
s_test <- subset(dataset_voice, split = "TRUE")

head(s_test)

#--------------------------------- Correlation
#install.packages("corrplot")
library("corrplot")
M <- cor(dataset_voice)
head(M)
corrplot(M, method = "shade")

# Get most p-value heavy dimensions
threshold <- 0.5
ok <- apply(abs(M) > threshold, 1, any)

ok

# Set variable with dimensions
cc = label ~ meanfreq + sd + median + Q25 + Q75 + IQR + skew + kurt + sp.ent + sfm + mode + centroid + meanfun + minfun + maxfun + meandom + mindom + maxdom + dfrange + modindx

#---------------------------------
#4. Linear Regression model
#---------------------------------

# Train the model

linearModel <- glm(cc,data = s_train, family = binomial())
summary(linearModel)

# Run model with test data

linearPred <- predict(linearModel, s_test, type = "response")
linearPred

#linearPred <- predict(linearModel, s_train, type = "response")
#linearPred

# Check models acc

con_matrix <- table(Actual_Value=s_train$label, Predicted_Value = linearRes > 0.5)
con_matrix

plot(linearPred, cc$residuals, xlab ="Predicted Values", ylab = "Error")

#---------------------------------
#5. Logistic Regression model
#---------------------------------

# Train the model

logisticModel <- glm(cc, data = s_train, family = "binomial")
summary(logisticModel)

# Run model with test data

logisticRes <- predict(logisticModel, s_test, type = "response")
logisticRes

#logisticRes <- predict(logisticModel, s_train, type = "response")
#logisticRes

# Check models acc

con_matrix <- table(Actual_Value=s_train$label, Predicted_Value = logisticRes > 0.5)
con_matrix
