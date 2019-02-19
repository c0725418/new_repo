#-------------------------------------------------------------
#2019W-T3 BDM 3014 - Introduction to Artificial Intelligence 01
#-------------------------------------------------------------
#Lab 01 – Logistic Regression
#-------------------------------------------------------------
#Lucicarla Silva de Santana				C0724974
#Valeria Ferreira de Almada Nobrega		C0724858
#Rafael Andrade Da Conceicao			C0725132
#Tulio Fernandes						C0722179
#Javier Navarro Gonzalez 				C0725418
#-------------------------------------------------------------


library("caTools")

#---------------------------------
#1. Reading and printing data in CSV file.
#---------------------------------
# Load the dataset from: 

#dataset_voice = read.csv("C:/projetos/ai/voice.csv")
dataset_voice = read.csv("C:/Users/Eduardo Carneiro/Desktop/Lambton/T3/3014 - Introduction to Artificial Intelligence 01/Dataset/voice.csv")
head(dataset_voice)
dim(dataset_voice)

#---------------------------------
#2. Data munging
#---------------------------------

dataset_voice$label <- ifelse(dataset_voice$label=="male",TRUE,FALSE)
# Categorical variables
#dataset_voice$label <- as.factor(dataset_voice$label)

#---------------------------------
#3 Split the data
#---------------------------------

split <- sample.split(dataset_voice, SplitRatio = 0.8)
split

s_train <- subset(dataset_voice, split == "TRUE")
s_test <- subset(dataset_voice, split == "FALSE")

head(s_train)
head(s_test)

#---------------------------------
#4. Fit Logistic Regression Model and accuracy
#---------------------------------

# 4.1 This model uses all columns from the dataset

## Set all variables with dimensions
full_dim = label ~ meanfreq + sd + median + Q25 + Q75 + IQR + skew + kurt + sp.ent + sfm + mode + centroid + meanfun + minfun + maxfun + meandom + mindom + maxdom + dfrange + modindx
full_dim

## Train the model
logisticModel_full <- glm(full_dim, data = s_train, family = "binomial")
summary(logisticModel_full)

## Run model with test data
logisticRes_full <- predict(logisticModel_full, s_test, type = "response")
head(logisticRes_full)

# 4.2 Check models acc
con_matrix_full <- table(Actual_Value = s_test$label, Predicted_value = logisticRes_full > 0.5)
con_matrix_full

# 4.3 Accuracy
## Acc. 96% -> All columns
(con_matrix_full[[1,1]] + con_matrix_full[[2,2]]) / sum(con_matrix_full)

#---------------------------------
#5. Correlation
#---------------------------------

#install.packages("corrplot")
library("corrplot")
M <- cor(dataset_voice[,-21])
head(M)

# Zero the diagonal intersections(they all have value 1)
diag(M) <- 0
corrplot(M, method = "shade")

# Get most p-value heavy dimensions
threshold <- 0.5
ok <- apply(abs(M) > threshold, 1, any)

ok

useful_vars = subset(ok, ok == TRUE)
useful_vars
remove_dim = names(subset(ok, ok == FALSE))
remove_dim
select_dim = as.formula(paste0("label~",paste(names(useful_vars),collapse = "+")))

select_dim


#---------------------------------
#6. Logistic Regression model
#---------------------------------

# 6.1 Remove unused dimensions
names(s_train)
s_train_select <- subset(s_train, select=-c(minfun, maxfun, mindom, modindx))
names(s_train_select)

s_train_manual <- subset(s_train, select=c(Q25, Q75, kurt, sp.ent, sfm, meanfun, label))
head(s_train_manual)

# 6.2 Train the model

## This model uses columns from the dataset automatically selected by their correlation score
logisticModel_select <- glm(select_dim, data = s_train_select, family = "binomial")
summary(logisticModel_select)

## This model uses columns from the dataset manually selected by their correlation score
logisticModel_manual <- glm(label ~ Q25 + Q75 + kurt + sp.ent + sfm + meanfun, data = s_train_manual, family = "binomial")
summary(logisticModel_select)

# 6.3 Run model with test data

## This model uses columns from the dataset automatically selected by their correlation score
logisticRes_select <- predict(logisticModel_select, s_test, type = "response")
head(logisticRes_select)

## This model uses columns from the dataset manually selected by their correlation score
logisticRes_manual <- predict(logisticModel_manual, s_test, type = "response")
head(logisticRes_manual)

# 6.4 Check models acc

## For model using columns from the dataset automatically selected by their correlation score
con_matrix_select <- table(Actual_Value=s_test$label, Predicted_value = logisticRes_select > 0.5)
con_matrix_select

## For model using columns from the dataset manually selected by their correlation score
con_matrix_manual <- table(Actual_Value=s_test$label, Predicted_value = logisticRes_manual > 0.5)
con_matrix_manual

# 6.5 Accuracy

## Acc. 96% -> Columns automatically selected
(con_matrix_select[[1,1]] + con_matrix_select[[2,2]]) / sum(con_matrix_select)

## Acc. 97% -> Columns manually selected
(con_matrix_manual[[1,1]] + con_matrix_manual[[2,2]]) / sum(con_matrix_manual)

