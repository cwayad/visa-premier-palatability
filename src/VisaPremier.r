#Lecture des donn?es
setwd("./data")
data = read.table('VisaPremier.txt',header = TRUE,na.strings = ".", stringsAsFactors = FALSE)
summary(data) #data[,47] c'est la variable ? expliquer y: pocession ou non de la carte primer visa
#preprocessing des donn?es
#1- elemination des variables inutiles
unused_data = names(data) %in% c("matricul","departem","sexe","ptvente","sitfamil","codeqlt","nbimpaye","csp","agemvt","cartevp","sexer")
data = data[!unused_data] #donn?es sans les colonnes inutiles pour l'etude concern?e
#Les clients sont ordonn?s, nous m?langons les lignes pour un traitement plus facile apr?s
data=data[sample(nrow(data)),]
inputs = data[!(names(data) %in% c("cartevpr"))] #Vecteurs des variables explicatives
output = data[,"cartevpr"] #Vecteur de la classe(2 classes) pocession ou non de la carte Visa Premier
View(inputs) # Nous remarquons que sur la colonne "nbpaiecb" il y 'en a des valeurs manquantes
#2- traitement des valeurs manquantes
# nous rempla?ons les valeurs manquante dans la colonne 31 = nbpaiecb par la moyenne de toute les valeurs de cette colonne
mean_nbpaiecb = ceiling(mean(inputs[,31],na.rm = TRUE))
inputs[,31][is.na(inputs[,31])]= mean_nbpaiecb

#Nous remarquons que les valeurs sont dans des intervalles tres diff?rents, 
#pour y palier, nous appliquons une normalisation
View(inputs)
inputs = scale(inputs, center = TRUE, scale = TRUE)

#concat?nation des inputs et le output
visaPremier= cbind(inputs[,], output[])
View(visaPremier)
#S?paration des donn?es d'apprentissage et test
visaPremier=visaPremier[,-c(28,29,30)] # a enlever car elle contienent les 0 par tout
train_visa = visaPremier[1:804,]
test_visa =  visaPremier[805:1073,]

#Maintenant que tout est pr?t, on peut appliquer les algos d'apprentissage supervis?.

########################################################################################
#                                  Apprentissage & test                                #
########################################################################################
train_visa= data.frame(train_visa)
test_visa= data.frame(test_visa)

#*******************************Regression Logistique**********************************#
#Apprentissage sur l'ensemble de Visa_train
library(MASS)
Visa.logReg<-glm(train_visa[,34] ~ ., data=train_visa[,-34])
#Pr?diction
Visa.logReg.pred = predict(Visa.logReg, newdata=test_visa[,-34],type='response')
#Selon le seuil, 
Visa.logReg.pred <- ifelse(Visa.logReg.pred > 0.5,1,0)

#Calcul de la pr?cision
Visa.logReg.erreurClass = mean(Visa.logReg.pred != test_visa[,34])
Visa.logReg.accurcy = 1 - Visa.logReg.erreurClass
print(paste("Pr?cision Logistic Regression =",
            round(Visa.logReg.accurcy, digits = 2)*100,"%"))

#La regression logistique donne une pr?cision de 85% :) c pas mal!
library(ROCR)
logReg_roc = prediction(as.numeric(Visa.logReg.pred),as.numeric(test_visa[,34]))
plot(performance(logReg_roc, "tpr", "fpr"),main = "Courbe ROC de LDA")
#*****************************************LDA*****************************************#
library(MASS)
#Construction de discrimination lin?aire
Visa.lda <- lda(train_visa[,-34], train_visa[,34]) 
#Prediction selon le mod?le lin?aire
Visa.lda.pred <- predict(Visa.lda,test_visa[,-34]) 
#Table de confusion
table(Visa.lda.pred$class,test_visa[,34]) 

#Calcul de la p?cision que la lDA donne
Visa.lda.erreurClass <- mean(Visa.lda.pred$class != test_visa[,34])
Visa.lda.accurcy = 1 - Visa.lda.erreurClass
print(paste("Pr?cision LDA =",round(Visa.lda.accurcy, digits = 2)*100,"%"))
#LDA donne une pr?cision de 85% ey ey
#ROC
lda_roc = prediction(as.numeric(Visa.lda.pred$class),as.numeric(test_visa[,34]))
plot(performance(lda_roc, "tpr", "fpr"),main = "Courbe ROC de LDA")
#*****************************************KNN******************************************#
library(class)
#Pour trouver de meilleure performance, nous devrions tester l'algorithme KNN pour 
#diff?rentes valeurs de K, nous ne sauvegardons que le k donnant la meilleure pr?cision.
k.optim = 0
Visa.KNN.accuracy = 0
for (i in 1:10) {
  Visa.KNN = knn(train_visa[,-34], test_visa[,-34], cl = train_visa[,34] , k = i)
  tmp = 1 - ( sum(Visa.KNN != test_visa[,34]) / length(test_visa[,34]) )
  if(tmp > Visa.KNN.accuracy){
    Visa.KNN.accuracy = tmp
    k.optim = i
  }
}

print(paste("Pr?cision ",k.optim ,"NN =",round(Visa.KNN.accuracy, digits = 2)*100,"%"))
#La meilleur valeur de K (allant de 1:10 ou m?me 1:20) est 4
#La pr?cision de 3NN est 83% 
#ROC
knn_roc = prediction(as.numeric(Visa.KNN),as.numeric(test_visa[,34]))
plot(performance(knn_roc, "tpr", "fpr"),main = "Courbe ROC de NN")

#*****************************Classificateur Baysien Naif*******************************#

library(e1071)
# apprentissage sur le training set
Visa.NaiveB = naiveBayes(as.factor(train_visa[,34]) ~ ., data = train_visa[,-34])
#Prediction sur l'?chantillon test
Visa.NaiveB.pred = predict(Visa.NaiveB, test_visa[,-34]) 
#Table de confusion
table(Visa.NaiveB.pred,test_visa[,34])
#Calcul de la pr?cision
Visa.NaiveB.erreurClass <- mean(Visa.NaiveB.pred != test_visa[,34])
Visa.NaiveB.accurcy = 1 - Visa.NaiveB.erreurClass
print(paste("Pr?cision Naive Bayes =",round(Visa.NaiveB.accurcy, digits = 2)*100,"%"))
# Le score obtenu est de 80% 
#ROC
NaivB_roc = prediction(as.numeric(Visa.NaiveB.pred),as.numeric(test_visa[,34]))
plot(performance(NaivB_roc, "tpr", "fpr"),main = "Courbe ROC du Bayes Naive")

#*****************************************QDA*****************************************#
#Construction de discrimination quadratique
Visa.qda <- qda(train_visa[,-34], train_visa[,34]) 
#Prediction selon le mod?le quadratique
Visa.qda.pred <- predict(Visa.qda,Visa_test[,-3]) 
#Table de confusion
table(Visa.qda.pred$class, Visa_test[,3]) 
#Calcul de la pr?cision
Visa.qda.erreurClass <- mean(Visa.qda.pred$class != Visa_test[,3])
Visa.qda.accurcy = 1 - Visa.qda.erreurClass
print(paste("Pr?cision QDA =",round(Visa.qda.accurcy, digits = 2)*100,"%"))

#Contrairement ? ce qu'on aurait pu penser la QDA ne donne pas de tr?s bon r?sultats
#Seulement 12% de pr?cision (mieux que LDA mais moins bonne que KNN et Naive Bayes :/ 


#*****************************************SVM*****************************************#

#D?finition des kernls n?cissaires au SVM
kernels = c("linear","polynomial","radial","sigmoid")
#La fonction tune ?stime les param?tres
Visa.svmTune = tune(svm, train.x = train_visa[,-34], train.y = train_visa[,34], 
                           validation.x = test_visa[,-34], validation.y = test_visa[,34], 
                           ranges = list(kernel = kernels))

# pr?cision du meilleur mod?le
Visa.svm.accurcy = 1 - Visa.svmTune$best.performance 
print(paste("Pr?cision SVM (",Visa.svmTune$best.parameters$kernel,") =",round(Visa.svm.accurcy, digits = 2)*100,"%"))
#Avec le kernel gaussien radial, nous obtenons un taux de pr?cision sup?rieur ? 88% :D 
#ROC
svm_roc = predict(as.numeric(Visa.svmTune),as.numeric(test_visa[,34]))
plot(svm_roc,"tpr","fpr",main = "Courbe ROC de SVM") 





