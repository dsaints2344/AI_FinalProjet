library(neuralnet)
library(mlbench)
library(dplyr)
library(caret)
library(ROCR)
library(pROC)
library(ggplot2)

# A) Create neural network models
# SELECT INSIGHT FEATURES

data(BreastCancer)

rawData = BreastCancer

# Changing values of BreastCander data class property for 0 and 1 in order to handle data easier

rawData$Class<-recode(rawData$Class, 'benign'=0, 'malignant'=1)

# B) 60% : 40% training sets

sampleSize <- floor(0.60 * nrow(rawData))

##set the seeder of the random generator in order to keep data reproductible
set.seed(127)
train_ind <- sample(seq_len(nrow(rawData)), size = sampleSize)
train <- rawData[train_ind, ]
test <- rawData[-train_ind, ]


# Converting data into training data.
Class = as.matrix((train$Class))
Cl.thickness = as.numeric((train)$Cl.thickness)
Cell.size = as.numeric((train)$Cell.size)
Cell.shape = as.numeric((train)$Cell.shape)


df=data.frame(Cl.thickness,Cell.size,Cell.shape,Class)

# C) Training model

breastCancerNN=neuralnet(Class~Cl.thickness+Cell.size+Cell.shape,data=df, hidden=3,act.fct = "logistic",
             linear.output = FALSE)



# D) Plotting neural network
plot(breastCancerNN)




## E) Neural network prediction

# data convertions for data
Cl.thickness = as.numeric((test)$Cl.thickness)
Cell.size = as.numeric((test)$Cell.size)
Cell.shape = as.numeric((test)$Cell.shape)


dfTest=data.frame(Cl.thickness,Cell.size,Cell.shape)


Predict <- neuralnet::compute(breastCancerNN,dfTest)
Predict$net.result


# Converting probabilities into binary classes setting threshold level 0.5
prob <- Predict$net.result
pred <- ifelse(prob>0.5, 1, 0)


# Confusion matrix

statistics <- confusionMatrix(as.factor(pred),as.factor(test$Class))
statistics

# F) ROCR


roc_obj <- pROC::roc(test$Class,as.numeric(pred))
auc(roc_obj)
plot.roc(roc_obj,add=FALSE,reuse.auc=TRUE,aces=TRUE,legacy.axes = FALSE,print.auc = TRUE)


# G Analisis de Performance del Modelo
# El modelo muestra predecir correctamente el 95.71% de los casos (268 de 280),
# En este sentido, el modelo prueba ser factible para ser utilizado en la predicciÃ³n preliminar de 
# celulas cancerigenas.
# Entre los indicadores de performance que muestran ser mas significativos se encuentran: 
# -Accuracy : 95.71% : Alto, por encima de la expectativa (85%)
# - Ratio de Falsos Negativos: 2.88% : Muy Bajo
# - F1 Score: 96.53% : Por Encima de la expectativa (80%)

# H) Trying to predict aggregated values
Cl.thickness = as.numeric((train)$Cl.thickness)
Cell.size = as.numeric((train)$Cell.size)
Cell.shape = as.numeric((train)$Cell.shape)
Epith.c.size = as.numeric((train)$Epith.c.size)
Marg.adhesion = as.numeric((train)$Marg.adhesion)

df=data.frame(Cl.thickness,Cell.size,Cell.shape,Epith.c.size,Marg.adhesion,Class)

breastCancerNNWithMoreVariables=neuralnet(Class~Cl.thickness+Cell.size+Cell.shape+Epith.c.size+Marg.adhesion,data=df, hidden=3,act.fct = "logistic",
                         linear.output = FALSE)


#prediction

Cl.thickness = as.numeric((test)$Cl.thickness)
Cell.size = as.numeric((test)$Cell.size)
Cell.shape = as.numeric((test)$Cell.shape)
Epith.c.size = as.numeric((test)$Epith.c.size)
Marg.adhesion = as.numeric((test)$Marg.adhesion)

dfTest=data.frame(Cl.thickness,Cell.size,Cell.shape,Epith.c.size,Marg.adhesion)


PredictMoreVariables <- neuralnet::compute(breastCancerNNWithMoreVariables,dfTest)


# Converting probabilities into binary classes setting threshold level 0.5
probWithMoreVariables <- PredictMoreVariables$net.result
predWithMoreVariables <- ifelse(probWithMoreVariables>0.5, 1, 0)


statisticsWithMoreVariables <- confusionMatrix(as.factor(predWithMoreVariables),as.factor(test$Class))
statisticsWithMoreVariables


# I) LOGISTIC MODEL


Cl.thickness = as.numeric((train)$Cl.thickness)
Cell.size = as.numeric((train)$Cell.size)
Cell.shape = as.numeric((train)$Cell.shape)
Epith.c.size = as.numeric((train)$Epith.c.size)
Marg.adhesion = as.numeric((train)$Marg.adhesion)


# Build Logistic Model using training data
logitmod <- glm(Class ~ Cl.thickness + Cell.size + Cell.shape, family = "binomial", train)

summary(logitmod)
anova(logitmod)

predLogistic <- predict(logitmod, newdata = test, type = "response")
predLogistic <- ifelse(predLogistic > 0.5, 1, 0)

# Accuracy and contigency analysis
statisticsLogistic <- confusionMatrix(factor(predLogistic,levels=c(0,1)),as.factor(test$Class))
statisticsLogistic

#PLOT ROC CURVE- PERFORMANCE


roc_obj_log <- pROC::roc(test$Class,as.numeric(predLogistic))
auc(roc_obj_log)
plot.roc(roc_obj_log,add=FALSE,reuse.auc=TRUE,aces=TRUE,legacy.axes = FALSE,print.auc = TRUE)


# J, K )
## COMENTARIO : El neural network parece ser mejor en terminos del ROC, Accuracy ( NN: 95.71 , 94.29) & F1 score (96.53 , 95.51)
## aunque la diferencia no parece ser demasiado significativa.
## En este sentido, viendo los resultados obtenidos por las tablas de contingencia con los mismos sets de datos,
## Entendemos que es correcto decir que el modelo de redes neuronales funciona mejor en este caso.

# L) Trying to predict adding Epith.c.size & Marg.adhesion


# Build Logistic Model using training data

Cl.thickness = as.numeric((train)$Cl.thickness)
Cell.size = as.numeric((train)$Cell.size)
Cell.shape = as.numeric((train)$Cell.shape)
Epith.c.size = as.numeric((train)$Epith.c.size)
Marg.adhesion = as.numeric((train)$Marg.adhesion)

logitmodWithMoreVariables <- glm(Class ~ Cl.thickness + Cell.size + Cell.shape + Epith.c.size + Marg.adhesion, family = "binomial", train)

predLogisticWithMoreVariables <- predict(logitmodWithMoreVariables, newdata = test, type = "response")
predLogisticWithMoreVariables <- ifelse(predLogisticWithMoreVariables > 0.5, 1, 0)

# Accuracy and contigency analysis
statisticsLogisticWithMoreVariables <- confusionMatrix(as.factor(predLogisticWithMoreVariables),as.factor(test$Class))
statisticsLogisticWithMoreVariables
