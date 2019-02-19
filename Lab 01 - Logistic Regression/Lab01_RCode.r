#1.1 Reading and printing data in CSV file.

dataset_voice= read.csv("D:/Lambton College/Term 3/Introduction to Artificial Intelligence/Labs/Lab 01/voice.csv")
dataset_voice$label <- ifelse(dataset_voice$label=="male",1,0)

dataset_voice
dataset_voice.ncol()
#slicing our data set
X <- dataset_voice[,-ncol(dataset_voice)] 
X
y <- dataset_voice$label
y


#Sampling : Divide the data into Train and Test datasets
set.seed(777)

index=sample(1:nrow(X),round(nrow(X)*.6))

train=X[index,]
test=X[-index,]

#have to convert it to data that can be read by our model

names(train)
num_var <- c("meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid","meanfun","minfun","maxfun" ,"meandom" ,"mindom" , "maxdom", "dfrange","modindx" )

#4. Split the data set 
useful_vars <- c()
for(var in num_var){
  tt <- t.test(train[,var]~train$label)
  if (tt$p.value < 0.05){
    print(var)
    print(tt$p.value)
    useful_vars=c(useful_vars,var)
  }
}  



#fml = as.formula(paste0("label~",paste(useful_vars,collapse = "+")))

#fit <- glm(fml,data = train, family = binomial())
#summary(fit)

fit2 <- glm(label~Q25+kurt+sp.ent+sfm+meanfun+minfun,data = train, family = binomial())
summary(fit2)

fit3 <- glm(label~median+kurt+meanfreq+sd+sp.ent+minfun+sfm+Q25+meanfun,data = train, family = binomial())
summary(fit3)

#model.step <- stepAIC(fit)
#summary(model.step)

train$pred <- predict(fit3,type='response')
train$pred_label=ifelse(train$pred>0.05,1,0) #sets the threshold
hist(train$pred)

table(Pred=train$pred_label,Act=train$label)

library(ROCR)
pred <- prediction(train$pred,train$label)
roc.perf=performance(pred,measure = "tpr",x.measure="fpr")
plot(roc.perf)
roc.perf@alpha.values

auc = performance(pred,measure="auc")
auc@y.values[[1]]

test$pred <- predict(fit3,type = 'response',newdata = test)
test$pred_label <- ifelse(test$pred>0.05,1,0) 

table(Pred=test$pred_label,Act=test$label)
