# Redirection vers le bon path
setwd("./data")

#Chargement des donn�es
aggregation <- read.table("Aggregation.txt", header = T,sep="\t")


#Aggregation
dim(aggregation)
plot(aggregation$X15.55, aggregation$X28.65, col= aggregation$X2, 
    pch=21,bg=c("blue","yellow","red","gray","green", "black", "magenta")
    [as.numeric(aggregation$X2)], main="Aggregation")

dim(aggregation)
#aggregation[,-3] : Les variables de aggregation
#aggregation[,3] : les classes (2 classes) de aggregation

#Nous eclatons aggregation en 2 �chantillons 
#Apprentissage : 89% aggregation ==>  700 ind pour le training set
#Test : 11% aggregation ==> 87 pour test set

aggregation_train = aggregation[1:87,]
aggregation_test = aggregation[88:787,]

#V�rification qu'on a bien 61 pour le test et 250 pour l'apprentissage
dim(aggregation_train)
dim(aggregation_test)

########################################################################################
#                                  Apprentissage & test                                #
########################################################################################


#*******************************Regression Logistique**********************************#
#Apprentissage sur l'ensemble de aggregation_train
aggregation.logReg<- glm(aggregation_train[,3] ~.,data=aggregation_train[,-3])
#Pr�diction
aggregation.logReg.pred <- predict(aggregation.logReg,newdata=aggregation_test[,-3],type='response')
#Selon le seuil, 
aggregation.logReg.pred <- ifelse(aggregation.logReg.pred > 0.5,1,0)
#Calcul de la pr�cision
aggregation.logReg.erreurClass = mean(aggregation.logReg.pred != aggregation_test[,3])
aggregation.logReg.accurcy = 1 - aggregation.logReg.erreurClass
print(paste("Pr�cision Logistic Regression =",
            round(aggregation.logReg.accurcy, digits = 2)*100,"%"))

#La regression logistique donne une pr�cision de 6% :(

#*****************************************LDA*****************************************#
library(MASS)
#Construction de discrimination lin�aire
aggregation.lda <- lda(aggregation_train[,-3], aggregation_train[,3] ) 
#Prediction selon le mod�le lin�aire
aggregation.lda.pred <- predict(aggregation.lda,aggregation_test[,-3]) 
#Table de confusion
table(aggregation.lda.pred$class,aggregation_test[,3]) 

#Calcul de la p�cision que la lDA donne
aggregation.lda.erreurClass <- mean(aggregation.lda.pred$class != aggregation_test[,3])
aggregation.lda.accurcy = 1 - aggregation.lda.erreurClass
print(paste("Pr�cision LDA =",round(aggregation.lda.accurcy, digits = 2)*100,"%"))
#LDA donne une pr�cision de 36% :'( 

#*****************************************KNN******************************************#
library(class)
#Pour trouver de meilleure performance, nous devrions tester l'algorithme KNN pour 
#diff�rentes valeurs de K, nous ne sauvegardons que le k donnant la meilleure pr�cision.
k.optim = 0
aggregation.KNN.accuracy = 0
for (i in 1:10) {
  aggregation.KNN = knn(aggregation_train[,-3], aggregation_test[,-3], cl = aggregation_train[,3] , k = i)
  tmp = 1 - ( sum(aggregation.KNN != aggregation_test[,3]) / length(aggregation_test[,3]) )
  if(tmp > aggregation.KNN.accuracy){
    aggregation.KNN.accuracy = tmp
    k.optim = i
  }
}

print(paste("Pr�cision ",k.optim ,"NN =",round(aggregation.KNN.accuracy, digits = 2)*100,"%"))
#La meilleur valeur de K (allant de 1:10 ou m�me 1:20) est 4
#La pr�cision de 4NN est 92% meilleure que celle de la LDA; :)


#*****************************Classificateur Baysien Naif*******************************#

library(e1071)
# apprentissage sur le training set
aggregation.NaiveB = naiveBayes(as.factor(aggregation_train[,3]) ~ ., data = aggregation_train[,-3])
#Prediction sur l'�chantillon test
aggregation.NaiveB.pred = predict(aggregation.NaiveB, aggregation_test[,-3]) 
#Table de confusion
table(aggregation.NaiveB.pred,aggregation_test[,3])
#Calcul de la pr�cision
aggregation.NaiveB.erreurClass <- mean(aggregation.NaiveB.pred != aggregation_test[,3])
aggregation.NaiveB.accurcy = 1 - aggregation.NaiveB.erreurClass
print(paste("Pr�cision Naive Bayes =",round(aggregation.NaiveB.accurcy, digits = 2)*100,"%"))
# Le score obtenu est de 67% mieux que la LDA mais moins bon que KNN :/


#*****************************************QDA*****************************************#
#Construction de discrimination quadratique
aggregation.qda <- qda(aggregation_train[,-3], aggregation_train[,3]) 
#Prediction selon le mod�le quadratique
aggregation.qda.pred <- predict(aggregation.qda,aggregation_test[,-3]) 
#Table de confusion
table(aggregation.qda.pred$class, aggregation_test[,3]) 
#Calcul de la pr�cision
aggregation.qda.erreurClass <- mean(aggregation.qda.pred$class != aggregation_test[,3])
aggregation.qda.accurcy = 1 - aggregation.qda.erreurClass
print(paste("Pr�cision QDA =",round(aggregation.qda.accurcy, digits = 2)*100,"%"))

#Contrairement � ce qu'on aurait pu penser la QDA ne donne pas de tr�s bon r�sultats
#Seulement 12% de pr�cision (mieux que LDA mais moins bonne que KNN et Naive Bayes :/ 


#*****************************************SVM*****************************************#
#D�finition des kernls n�cissaires au SVM
kernels = c("linear","polynomial","radial","sigmoid")
#La fonction tune �stime les param�tres
aggregation.svmTune = tune(svm, train.x = aggregation_train[,-3], train.y = aggregation_train[,3], 
                     validation.x = aggregation_test[,-3], validation.y = aggregation_test[,3], 
                     ranges = list(kernel = kernels))

# pr�cision du meilleur mod�le
aggregation.svm.accurcy = 1 - aggregation.svmTune$best.performance 
print(paste("Pr�cision SVM (",aggregation.svmTune$best.parameters$kernel,") =",
            round(aggregation.svm.accurcy, digits = 2)*100,"%"))
#Avec le kernel gaussien radial, nous obtenons un taux de pr�cision sup�rieur � 99% :D 
