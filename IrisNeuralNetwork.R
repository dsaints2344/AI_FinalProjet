library(neuralnet)
library(dplyr)
library(caret)
library(ROCR)
library(pROC)
library(ggplot2)
library(nnet)

data(iris)

rawData = iris

#convertimos las clases a numeros
rawData$Species<-recode(rawData$Species, 'setosa'=1, 'virginica'=2,'versicolor'=3)

#creamos un bit por cada clase (el que este encendido pertenece a esa clase)
encodedData <- cbind(rawData[,1:4], class.ind(as.factor(rawData$Species)))
names(encodedData) <- c(names(rawData)[1:4],"setosa","virginica","versicolor")

# B) 67% TRAINING 33% TEST

smp_size <- floor(0.67 * nrow(encodedData))

## set the seed to make your partition reproducible
set.seed(127)
train_ind <- sample(seq_len(nrow(encodedData)), size = smp_size)

train <- encodedData[train_ind, ]
test <- encodedData[-train_ind, ]



# CREATE NEURAL NETWORK


setosa = as.matrix((train$setosa))
virginica = as.matrix((train$virginica))
versicolor = as.matrix((train$versicolor))
Sepal.Length = as.numeric((train)$Sepal.Length)
Sepal.Width = as.numeric((train)$Sepal.Width)
Petal.Length = as.numeric((train)$Petal.Length)
Petal.Width = as.numeric((train)$Petal.Width)


df=data.frame(Sepal.Length,Sepal.Width,Petal.Length,Petal.Width,setosa,virginica,versicolor)

# C) TRAIN MODEL WITH TRAINING DATA

irisNN=neuralnet(setosa+virginica+versicolor~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width,data=df, hidden=3,act.fct = "logistic",
                         linear.output = FALSE)

# D) PLOT NEURAL NETWORK
plot(irisNN)

## E) Prediction using neural network

# MAKE CONVERSIONS FOR TEST
Sepal.Length = as.numeric((test)$Sepal.Length)
Sepal.Width = as.numeric((test)$Sepal.Width)
Petal.Length = as.numeric((test)$Petal.Length)
Petal.Width = as.numeric((test)$Petal.Width)


dfTest=data.frame(Sepal.Length,Sepal.Width,Petal.Length,Petal.Width)


Predict <- neuralnet::compute(irisNN,dfTest)


# Converting probabilities into binary classes setting threshold level 0.5
prob <- Predict$net.result
pred <- ifelse(prob>0.5, 1, 0)




# ESTADISTICAS : Matriz de ConfusiÃ³n
testClass = data.frame(test$setosa,test$virginica,test$versicolor)
names(testClass) <- c("setosa","virginica","versicolor")

pred = data.frame(pred)
names(pred) <- c("setosa","virginica","versicolor")


#convertir bits a clases (1 = setosa, 2 = virginica ,3 = versicolor)
test$Species[test$setosa == '1'] =1
test$Species[test$virginica == '1'] =2
test$Species[test$versicolor == '1'] =3


pred$Species[pred$setosa == '1'] =1
pred$Species[pred$virginica == '1'] =2
pred$Species[pred$versicolor == '1'] =3


#creando matriz de confusiÃ³n

statistics <- confusionMatrix(as.factor(pred$Species),as.factor(test$Species))
statistics



# F) ROCR


roc_obj <- pROC::multiclass.roc(test$Species,as.numeric(pred$Species))
auc(roc_obj)
plot.roc(roc_obj[['rocs']][[1]],add=FALSE,reuse.auc=TRUE,aces=TRUE,legacy.axes = FALSE,print.auc = TRUE)


# G Analisis de Performance del Modelo
# El modelo muestra predecir correctamente el 95.92% de los casos (47 de 50),
# prediciendo correctamente el 100% de las setosas y versicolores y equivocandose unicamente al
# deducir erroneamente algunas virginicas.
# En este sentido, el modelo prueba ser factible para ser utilizado en la predicciÃ³n preliminar de 
# especies.
# Entre los indicadores de performance que muestran ser mas significativos se encuentran: 
# -Accuracy : 95.92% : Alto, por encima de la expectativa (85%)
# -Ratio de no informaciÃ³n (clases que no pudieron ser predecidas) : 3.88% : Muy Bajo
# - Especificidad: Por Encima de 93% para cada clase, lo que muestra que los features utilizados son muy
# buenos indicadores del tipo de planta a predecir.


# H intentar reduciendo las variables 


setosa = as.matrix((train$setosa))
virginica = as.matrix((train$virginica))
versicolor = as.matrix((train$versicolor))
Sepal.Length = as.numeric((train)$Sepal.Length)
Sepal.Width = as.numeric((train)$Sepal.Width)
Petal.Length = as.numeric((train)$Petal.Length)
Petal.Width = as.numeric((train)$Petal.Width)


## NEURAL NETWORKS
# sin Petal.Width (COMENTADO PORQUE NO CONVERGE)

#df=data.frame(Sepal.Length,Sepal.Width,Petal.Length,setosa,virginica,versicolor)
#irisNNWOPetalWidth=neuralnet(setosa+virginica+versicolor~Sepal.Length+Sepal.Width+Petal.Length,data=df, hidden=3,act.fct = "logistic",
#                 linear.output = FALSE)


# sin Petal.Length

df=data.frame(Sepal.Length,Sepal.Width,Petal.Width,setosa,virginica,versicolor)
irisNNWOPetalLength=neuralnet(setosa+virginica+versicolor~Sepal.Length+Sepal.Width+Petal.Width,data=df, hidden=3,act.fct = "logistic",
                 linear.output = FALSE)


# sin Sepal.Width

df=data.frame(Sepal.Length,Petal.Length,Petal.Width,setosa,virginica,versicolor)
irisNNWOSepalWidth=neuralnet(setosa+virginica+versicolor~Sepal.Length+Petal.Length+Petal.Width,data=df, hidden=3,act.fct = "logistic",
                 linear.output = FALSE)

# sin Sepal.Length

df=data.frame(Sepal.Width,Petal.Length,Petal.Width,setosa,virginica,versicolor)
irisNNWOSepalLength=neuralnet(setosa+virginica+versicolor~+Sepal.Width+Petal.Length+Petal.Width,data=df, hidden=3,act.fct = "logistic",
                 linear.output = FALSE)


##PREDICTIONS

Sepal.Length = as.numeric((test)$Sepal.Length)
Sepal.Width = as.numeric((test)$Sepal.Width)
Petal.Length = as.numeric((test)$Petal.Length)
Petal.Width = as.numeric((test)$Petal.Width)


# sin Sepal.Width (COMENTADO PORQUE NO CONVERGE)
#dfTestwOPetalWidth=data.frame(Sepal.Length,Sepal.Width,Petal.Length)
#PredictWOPetalWidth <- neuralnet::compute(irisNNWOPetalWidth,dfTestwOPetalWidth)
#probWOPetalWidth <- PredictWOPetalWidth$net.result
#predWOPetalWidth <- ifelse(probWOPetalWidth>0.5, 1, 0)
#predWOPetalWidth = data.frame(predWOPetalWidth)
#names(predWOPetalWidth) <- c("setosa","virginica","versicolor")
#predWOPetalWidth$Species[predWOPetalWidth$setosa == '1'] =1
#predWOPetalWidth$Species[predWOPetalWidth$virginica == '1'] =2
#predWOPetalWidth$Species[predWOPetalWidth$versicolor == '1'] =3
#statisticsWOPetalWidth <- confusionMatrix(as.factor(predWOPetalWidth$Species),as.factor(test$Species))
#statisticsWOPetalWidth



# sin Petal.Length
dfTestwOPetalLength=data.frame(Sepal.Length,Sepal.Width,Petal.Width)
PredictWOPetalLength <- neuralnet::compute(irisNNWOPetalLength,dfTestwOPetalLength)
probWOPetalLength <- PredictWOPetalLength$net.result
predWOPetalLength <- ifelse(probWOPetalLength>0.5, 1, 0)
predWOPetalLength = data.frame(predWOPetalLength)
names(predWOPetalLength) <- c("setosa","virginica","versicolor")
predWOPetalLength$Species[predWOPetalLength$setosa == '1'] =1
predWOPetalLength$Species[predWOPetalLength$virginica == '1'] =2
predWOPetalLength$Species[predWOPetalLength$versicolor == '1'] =3
statisticsWOPetalLength <- confusionMatrix(as.factor(predWOPetalLength$Species),as.factor(test$Species))
statisticsWOPetalLength


# sin Sepal.Width
dfTestwOSepalWidth=data.frame(Sepal.Length,Petal.Length,Petal.Width)
PredictWOSepalWidth <- neuralnet::compute(irisNNWOSepalWidth,dfTestwOSepalWidth)
probWOSepalWidth <- PredictWOSepalWidth$net.result
predWOSepalWidth <- ifelse(probWOSepalWidth>0.5, 1, 0)
predWOSepalWidth = data.frame(predWOSepalWidth)
names(predWOSepalWidth) <- c("setosa","virginica","versicolor")
predWOSepalWidth$Species[predWOSepalWidth$setosa == '1'] =1
predWOSepalWidth$Species[predWOSepalWidth$virginica == '1'] =2
predWOSepalWidth$Species[predWOSepalWidth$versicolor == '1'] =3
statisticsWOSepalWidth <- confusionMatrix(as.factor(predWOSepalWidth$Species),as.factor(test$Species))
statisticsWOSepalWidth


# sin Sepal.Length
dfTestwOSepalLength=data.frame(Sepal.Width,Petal.Length,Petal.Width)
PredictWOSepalLength <- neuralnet::compute(irisNNWOSepalLength,dfTestwOSepalLength)
probWOSepalLength <- PredictWOSepalLength$net.result
predWOSepalLength <- ifelse(probWOSepalLength>0.5, 1, 0)
predWOSepalLength = data.frame(predWOSepalLength)
names(predWOSepalLength) <- c("setosa","virginica","versicolor")
predWOSepalLength$Species[predWOSepalLength$setosa == '1'] =1
predWOSepalLength$Species[predWOSepalLength$virginica == '1'] =2
predWOSepalLength$Species[predWOSepalLength$versicolor == '1'] =3
statisticsWOSepalLength <- confusionMatrix(as.factor(predWOSepalLength$Species),as.factor(test$Species))
statisticsWOSepalLength
