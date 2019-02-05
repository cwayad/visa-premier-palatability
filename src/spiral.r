# Redirection vers le bon path
setwd("./data")

#Chargement des donn?es
spiral <- read.table("spiral.txt",header = T, sep = "\t", fill = T)

#spiral
dim(spiral)
plot(spiral$X31.95, spiral$X7.95, col= spiral$X3, pch=21,bg=c("blue", "magenta", "green")[as.numeric(spiral$X3)], main="spiral")

#spiral[,-3] : Les variables de spiral
#spiral[,3] : les classes (2 classes) de spiral

#Nous eclatons spiral en 2 ?chantillons (? 1/3 de la classe 1, 1/3 de la classe 2 et 1/3 de la classe 3)
#Apprentissage : 80% spiral ==>  250 ind pour le training set
#Test : 20% spiral ==> 61 pour test set

spiral_train = rbind(spiral[21:105,],spiral[126:206,], spiral[228:311, ])
spiral_test = rbind(spiral[1:20,], spiral[106:125,], spiral[207:227, ])

#V?rification qu'on a bien 61 pour le test et 250 pour l'apprentissage
dim(spiral_train)
dim(spiral_test)

#La pr?cision, et la sp?cificit? semblent ?tre des bonnes mesures pour ?sitmer les 
#m?thodes sur ce jeu de donn?es
#Le nombre d'individu par classe est assez similaire

########################################################################################
#                                  Apprentissage & test                                #
########################################################################################


#*******************************Regression Logistique**********************************#
#Apprentissage sur l'ensemble de spiral_train
spiral.logReg<- glm(spiral_train[,3] ~.,data=spiral_train[,-3])
#Pr?diction
spiral.logReg.pred <- predict(spiral.logReg,newdata=spiral_test[,-3],type='response')
#Selon le seuil, 
spiral.logReg.pred <- ifelse(spiral.logReg.pred > 0.5,1,0)
#Calcul de la pr?cision
spiral.logReg.erreurClass = mean(spiral.logReg.pred != spiral_test[,3])
spiral.logReg.accurcy = 1 - spiral.logReg.erreurClass
print(paste("Pr?cision Logistic Regression =",
            round(spiral.logReg.accurcy, digits = 2)*100,"%"))

#La regression logistique donne une pr?cision de 33% :(

#*****************************************LDA*****************************************#
library(MASS)
#Construction de discrimination lin?aire
spiral.lda <- lda(spiral_train[,-3], spiral_train[,3] ) 
#Prediction selon le mod?le lin?aire
spiral.lda.pred <- predict(spiral.lda,spiral_test[,-3]) 
#Table de confusion
table(spiral.lda.pred$class,spiral_test[,3]) 

#Calcul de la p?cision que la lDA donne
spiral.lda.erreurClass <- mean(spiral.lda.pred$class != spiral_test[,3])
spiral.lda.accurcy = 1 - spiral.lda.erreurClass
print(paste("Pr?cision LDA =",round(spiral.lda.accurcy, digits = 2)*100,"%"))
#LDA donne une pr?cision de 0% :'(, si t'as dis je calcule pas ?a aurait ?t? mieux :/

#*****************************************KNN******************************************#
library(class)
#Pour trouver de meilleure performance, nous devrions tester l'algorithme KNN pour 
#diff?rentes valeurs de K, nous ne sauvegardons que le k donnant la meilleure pr?cision.
k.optim = 0
spiral.KNN.accuracy = 0
for (i in 1:10) {
  spiral.KNN = knn(spiral_train[,-3], spiral_test[,-3], cl = spiral_train[,3] , k = i)
  tmp = 1 - ( sum(spiral.KNN != spiral_test[,3]) / length(spiral_test[,3]) )
  if(tmp > spiral.KNN.accuracy){
    spiral.KNN.accuracy = tmp
    k.optim = i
  }
}

print(paste("Pr?cision ",k.optim ,"NN =",round(spiral.KNN.accuracy, digits = 2)*100,"%"))
#La meilleur valeur de K (allant de 1:10 ou m?me 1:20) est 1, ce qui est logique car 
#les spirales sont loins l'une des autres !
#La pr?cision de 1NN est 21% , De?ue :( !


#*****************************Classificateur Baysien Naif*****************************#

library(e1071)
# apprentissage sur le training set
spiral.NaiveB = naiveBayes(as.factor(spiral_train[,3]) ~ ., data = spiral_train[,-3])
#Prediction sur l'?chantillon test
spiral.NaiveB.pred = predict(spiral.NaiveB, spiral_test[,-3]) 
#Table de confusion
table(spiral.NaiveB.pred,spiral_test[,3])
#Calcul de la pr?cision
spiral.NaiveB.erreurClass <- mean(spiral.NaiveB.pred != spiral_test[,3])
spiral.NaiveB.accurcy = 1 - spiral.NaiveB.erreurClass
print(paste("Pr?cision Naive Bayes =",round(spiral.NaiveB.accurcy, digits = 2)*100,"%"))
# Le score obtenu est de 0% (au moins toi t'es hon?te -_-)


#*****************************************QDA*****************************************#
#Construction de discrimination quadratique
spiral.qda <- qda(spiral_train[,-3], spiral_train[,3]) 
#Prediction selon le mod?le quadratique
spiral.qda.pred <- predict(spiral.qda,spiral_test[,-3]) 
#Table de confusion
table(spiral.qda.pred$class, spiral_test[,3]) 
#Calcul de la pr?cision
spiral.qda.erreurClass <- mean(spiral.qda.pred$class != spiral_test[,3])
spiral.qda.accurcy = 1 - spiral.qda.erreurClass
print(paste("Pr?cision QDA =",round(spiral.qda.accurcy, digits = 2)*100,"%"))

# 0% :/ 


#*****************************************SVM*****************************************#
#D?finition des kernls n?cissaires au SVM
kernels = c("linear","polynomial","radial","sigmoid")
#La fonction tune ?stime les param?tres
spiral.svmTune = tune(svm, train.x = spiral_train[,-3], train.y = spiral_train[,3], 
                     validation.x = spiral_test[,-3], validation.y = spiral_test[,3], 
                     ranges = list(kernel = kernels))

# pr?cision du meilleur mod?le
spiral.svm.accurcy = 1 - spiral.svmTune$best.performance 
print(paste("Pr?cision SVM (",spiral.svmTune$best.parameters$kernel,") =",
            round(spiral.svm.accurcy, digits = 2)*100,"%"))
#Avec le kernel gaussien radial, nous obtenons un taux de pr?cision sup?rieur ? 87% :D 
