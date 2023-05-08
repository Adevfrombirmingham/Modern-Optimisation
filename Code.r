## loading packages
library(readxl)
library(readr)
library(tidyverse)
library(genalg)
library(ggplot2)
library(janitor)
library(lubridate)
library(forecast)
library(GA)
library(UsingR)
library(reshape2)
library(xgboost)
library(stringr)
options(scipen = 100)

set.seed(22127806)

## Creating functions
source("functions.r")



## loading the data
df = read.csv("Dataset - Modern Optimisation - Sheet1.csv")

## cleaning column names
df = clean_names(df)



## cleaning the data: numeric data type + singularity
cols_to_parse = names(df)[4:ncol(df)]
cols_to_parse =  cols_to_parse[cols_to_parse != "units_ordered_b2b"]
cols_to_parse =  cols_to_parse[cols_to_parse != "total_order_items_b2b"]
cols_to_parse =  cols_to_parse[cols_to_parse != "units_refunded"]
cols_to_parse =  cols_to_parse[cols_to_parse != "feedback_received"]
cols_to_parse =  cols_to_parse[cols_to_parse != "negative_feedback_received"]
cols_to_parse =  cols_to_parse[cols_to_parse != "a_to_z_claims_granted"]

## removing % $ , from the data
df[,cols_to_parse] =  df[,cols_to_parse] %>%
  mutate_at(vars(cols_to_parse), ~ parse_number(.))



## creating Date
df$Date = as.Date(paste0("01-",df$month), format =  "%d-%b-%Y")

## extracting month and year out of date
df$month = month(df$Date)
df$year = year(df$Date)

## this are useless
removeList = c(
  "received_negative_feedback_rate",
  "a_to_z_claims_granted",
  "feedback_received",
  "negative_feedback_received",
  "received_negative_feedback_rate",
  "Date",
  "claims_amount",
  "budget_pacing",
  "stop_over_spend"
) 


Selected = names(df)[!names(df) %in% removeList]

## clean data
dfClean = df[,Selected]





### ===========================================================================





















### ===========================================================================

# library(ggcorrplot)
# ggcorrplot(cor(dfClean))

# RunsList = c(10,20,30,40,50)

# PopList = c(50,100,30,40,50)

Pop50 = runGA(noRuns = 30, problem = "feature",dfClean,populationSize = 50)
Pop100 = runGA(noRuns = 30, problem = "feature",dfClean,populationSize = 100)
Pop200 = runGA(noRuns = 30, problem = "feature",dfClean,populationSize = 200)

P1 = parseData(Pop50,2,30)
P2 = parseData(Pop100,2,30)
P3 = parseData(Pop200,2,30)

plotbars(P1,P2,P3, "pop = 20", "pop=50", "pop = 200")

## based on this plot run the runGA function again

Pop200 = runGA(noRuns = 30, problem = "feature",dfClean,populationSize = 200)



## GA results

SummaryGA = summary(myGA)

## getting columns: selected features
SolCols = SummaryGA$solution %>% as.data.frame()

MeltedSolCols = melt(SolCols, id = NULL)

## keeping the selected features
MeltedSolCols = MeltedSolCols %>%
  filter(value == 1) 

SelectedFeatures = unique(MeltedSolCols$variable) %>% as.character()

SelectedFeatures = append(SelectedFeatures,"total_sales")

dfCleanSelectedFeatures = dfClean[,SelectedFeatures]


selected_NSGA2 = append(selected_NSGA2,"total_sales")

dfCleanSelectedFeaturesNSGA = dfClean[,selected_NSGA2]





## last 5 for test
train_ind <- 6:25

## train test split
train <- dfCleanSelectedFeatures[train_ind, ]
test <- dfCleanSelectedFeatures[-train_ind, ]


## LR on feature selected data : GA
ModelLR_FeaturesSelected = lm(total_sales~.,train)

summary(ModelLR_FeaturesSelected)

test$Predicted_FeatureSelected_LR = predict(ModelLR_FeaturesSelected,test)

# check error metrics
accuracy(test$Predicted_FeatureSelected_LR,test$total_sales)

MAPE_LR_F = accuracy(test$Predicted_FeatureSelected_LR,test$total_sales)





## last 5 for test
train_ind <- 6:25

## train test split
train2 <- dfCleanSelectedFeaturesNSGA[train_ind, ]
test2 <- dfCleanSelectedFeaturesNSGA[-train_ind, ]


## LR on feature selected data : NSGAII
ModelLR_FeaturesSelectedNSGA = lm(total_sales~.,train2)

summary(ModelLR_FeaturesSelectedNSGA)

test$Predicted_FeatureSelected_LR_NSGA = predict(ModelLR_FeaturesSelectedNSGA,test2)

# check error metrics
accuracy(test$Predicted_FeatureSelected_LR_NSGA,test2$total_sales)

MAPE_LR_NSGA = accuracy(test$Predicted_FeatureSelected_LR_NSGA,test2$total_sales)




train_ind <- 6:25

train1 <- dfClean[train_ind, ]
test1 <- dfClean[-train_ind, ]


ModelLR_AllFeatures = lm(total_sales~.,train1)

summary(ModelLR_AllFeatures)


test$Predicted_AllFeatures_LR_GA = predict(ModelLR_AllFeatures,test1)


accuracy(test$Predicted_AllFeatures_LR_GA,test$total_sales)

MAPE_LR_All = accuracy(test$Predicted_AllFeatures_LR_GA,test$total_sales)





## xg boost all data ----

test_x = data.matrix(dfClean[1:5,names(dfClean)[names(dfClean) != "total_sales"]])
test_y = dfClean[1:5,"total_sales"]


#define predictor and response variables in training set
train_x = data.matrix(dfClean[6:25,names(dfClean)[names(dfClean) != "total_sales"]])
train_y = dfClean[6:25,"total_sales"]



#define final training and testing sets
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)



#defining a watchlist
watchlist = list(train=xgb_train, test=xgb_test)

#fit XGBoost model and display training and testing data at each iteartion
# model = xgb.train(data = xgb_train, max.depth = 30, watchlist=watchlist, nrounds = 100)

#define final model
model_xgboost = xgboost(data = xgb_train, max.depth = 10, nrounds = 100, verbose = 0)

summary(model_xgboost)


#use model to make predictions on test data
pred_AllData = predict(model_xgboost, xgb_test)


accuracy(pred_AllData,test$total_sales)

MAPE_XGB_All = accuracy(pred_AllData,test$total_sales)



### xg boost feature selected GA -----


test_x = data.matrix(dfCleanSelectedFeatures[1:5,names(dfCleanSelectedFeatures)[names(dfCleanSelectedFeatures) != "total_sales"]])
test_y = dfCleanSelectedFeatures[1:5,"total_sales"]


#define predictor and response variables in training set
train_x = data.matrix(dfCleanSelectedFeatures[6:25,names(dfCleanSelectedFeatures)[names(dfCleanSelectedFeatures) != "total_sales"]])
train_y = dfCleanSelectedFeatures[6:25,"total_sales"]



#define final training and testing sets
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)



#defining a watchlist
watchlist = list(train=xgb_train, test=xgb_test)

#fit XGBoost model and display training and testing data at each iteartion
# model = xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 100)

#define final model
model_xgboost = xgboost(data = xgb_train, max.depth = 3, nrounds = 86, verbose = 0)

summary(model_xgboost)


#use model to make predictions on test data
pred_FeatureSelected = predict(model_xgboost, xgb_test)


accuracy(pred_FeatureSelected,test$total_sales)
MAPE_XGB_F = accuracy(pred_FeatureSelected,test$total_sales)


### xg boost feature selected NSGA-----


test_x = data.matrix(dfCleanSelectedFeaturesNSGA[1:5,names(dfCleanSelectedFeaturesNSGA)[names(dfCleanSelectedFeaturesNSGA) != "total_sales"]])
test_y = dfCleanSelectedFeaturesNSGA[1:5,"total_sales"]


#define predictor and response variables in training set
train_x = data.matrix(dfCleanSelectedFeaturesNSGA[6:25,names(dfCleanSelectedFeaturesNSGA)[names(dfCleanSelectedFeaturesNSGA) != "total_sales"]])
train_y = dfCleanSelectedFeaturesNSGA[6:25,"total_sales"]



#define final training and testing sets
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)



#defining a watchlist
watchlist = list(train=xgb_train, test=xgb_test)

#fit XGBoost model and display training and testing data at each iteartion
# model = xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 100)

#define final model
model_xgboostNSGA = xgboost(data = xgb_train, max.depth = 3, nrounds = 86, verbose = 0)

summary(model_xgboostNSGA)


#use model to make predictions on test data
pred_FeatureSelectedNSGA = predict(model_xgboostNSGA, xgb_test)


accuracy(pred_FeatureSelectedNSGA,test$total_sales)
MAPE_XGB_NSGA = accuracy(pred_FeatureSelectedNSGA,test$total_sales)





## visualizing the results

## date for plot
test$Date = as.Date(paste0(test$year,"-",test$month,"-01"))
names(test)

test$pred_FeatureSelected_XGB_GA = pred_FeatureSelected
test$pred_FeatureSelected_XGB_NSGAII = pred_FeatureSelectedNSGA
test$pred_AllData_XGB = pred_AllData

names(test)
MeltedDataPlot = melt(test[,c("Date","total_sales","Predicted_FeatureSelected_LR","Predicted_FeatureSelected_LR_NSGA",
                              "Predicted_AllFeatures_LR_GA",
                              "pred_AllData_XGB","pred_FeatureSelected_XGB_NSGAII",
                              "pred_FeatureSelected_XGB_GA")], id = "Date")  

MeltedDataPlot$variable = as.character(MeltedDataPlot$variable)
MeltedDataPlot$variable[MeltedDataPlot$variable == "Predicted_AllFeatures_LR_GA"] = "Predicted_AllFeatures_LR"
MeltedDataPlot$variable[MeltedDataPlot$variable == "Predicted_FeatureSelected_LR"] = "Predicted_FeatureSelected_LR_GA"
MeltedDataPlot$variable[MeltedDataPlot$variable == "Predicted_FeatureSelected_LR_NSGA"] = "Predicted_FeatureSelected_LR_NSGAII"

# MeltedDataPlot$variable

ggplot(data = MeltedDataPlot, aes(x = Date, y = value, color = variable)) +
  geom_line() +
  geom_point()+
  ylab("Total Sales")+
  ggtitle("Comparison of Different Models")






MAPE_LR_F[5] ## GA
MAPE_LR_All[5]
MAPE_LR_NSGA[5]

MAPE_XGB_F[5] ## GA
MAPE_XGB_All[5]
MAPE_XGB_NSGA[5]


ResultsMAPE = data.frame(
  Features = c("All","All"),
  Model = c("LR","XGB"),
  MAPE = c(MAPE_LR_All[5],MAPE_XGB_All[5])
)



ggplot(ResultsMAPE, aes(x=Model, y=MAPE, fill=Features)) +
  geom_bar(stat="identity", position=position_dodge())+
  theme_minimal()


ResultsMAPE = data.frame(
  Features = c("All","All","GA","GA","NSGAII","NSGAII"),
  Model = c("LR","XGB","LR","XGB","LR","XGB"),
  MAPE = c(MAPE_LR_All[5],MAPE_XGB_All[5],MAPE_LR_F[5],MAPE_XGB_F[5],MAPE_LR_NSGA[5],MAPE_XGB_NSGA[5])
)

ggplot(ResultsMAPE, aes(x=Model, y=MAPE, fill=Features)) +
  geom_bar(stat="identity", position=position_dodge())+
  theme_minimal()


