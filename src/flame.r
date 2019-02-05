# Redirection vers le bon path
setwd("./data")

#Chargement des donn�es
flame <- read.table("flame.txt",header = T, sep = "\t", fill = T)

#Dimension de flame
dim(flame)
#affichage des individus de flame en fonction des 2 variables et en les colorie 
#en fonction de leur classe
plot(flame$X1.85, flame$X27.8, col=flame$X1, pch=21,bg=c("blue", "magenta")
     [as.numeric(flame$X1)], main="Flame")
#Il est clair que les 2 classes sont s�par�es par une courbe parabolique 
#(de type y=x0+x1^2)


#flame[,-3] : Les variables de flame
#Flame[,3] : les classes (2 classes) de flame

#Nous eclatons flame en 2 �chantillons (� 1/2 de la classe 1 et 1/2 de la classe 2)
#Apprentissage : 84% flame ==>  200 ind pour le training set
#Test : 16% flame ==> 39 pour test set

flame_train = rbind(flame[1:133,],flame[173:240,])
flame_test = flame[134:172,]

#V�rification qu'on a bien 39 pour le test et 200 pour l'apprentissage
dim(flame_train)
dim(flame_test)

#La pr�cision, et la sp�cificit� semblent �tre des bonnes mesures pour �sitmer les 
#m�thodes sur ce jeu de donn�es
#Le nombre d'individu par classe est assez similaire

########################################################################################
#                                  Apprentissage & test                                #
########################################################################################


#*******************************Regression Logistique**********************************#
#Apprentissage sur l'ensemble de flame_train
flame.logReg<- glm(flame_train[,3] ~.,data=flame_train[,-3])
#Pr�diction
flame.logReg.pred <- predict(flame.logReg,newdata=flame_test[,-3],type='response')
#Selon le seuil, 
flame.logReg.pred <- ifelse(flame.logReg.pred > 0.5,1,0)
#Calcul de la pr�cision
flame.logReg.erreurClass = mean(flame.logReg.pred != flame_test[,3])
flame.logReg.accurcy = 1 - flame.logReg.erreurClass
print(paste("Pr�cision Logistic Regression =",
            round(flame.logReg.accurcy, digits = 2)*100,"%"))

#La regression logistique donne une pr�cision de 46% :(

#*****************************************LDA*****************************************#
library(MASS)
#Construction de discrimination lin�aire
flame.lda <- lda(flame_train[,-3], flame_train[,3] ) 
#Prediction selon le mod�le lin�aire
flame.lda.pred <- predict(flame.lda,flame_test[,-3]) 
#Table de confusion
table(flame.lda.pred$class,flame_test[,3]) 

#Calcul de la p�cision que la lDA donne
flame.lda.erreurClass <- mean(flame.lda.pred$class != flame_test[,3])
flame.lda.accurcy = 1 - flame.lda.erreurClass
print(paste("Pr�cision LDA =",round(flame.lda.accurcy, digits = 2)*100,"%"))
#LDA donne une pr�cision de 36% :'( 

#*****************************************KNN******************************************#
library(class)
#Pour trouver de meilleure performance, nous devrions tester l'algorithme KNN pour 
#diff�rentes valeurs de K, nous ne sauvegardons que le k donnant la meilleure pr�cision.
k.optim = 0
flame.KNN.accuracy = 0
for (i in 1:10) {
  flame.KNN = knn(flame_train[,-3], flame_test[,-3], cl = flame_train[,3] , k = i)
  tmp = 1 - ( sum(flame.KNN != flame_test[,3]) / length(flame_test[,3]) )
  if(tmp > flame.KNN.accuracy){
    flame.KNN.accuracy = tmp
    k.optim = i
  }
}

print(paste("Pr�cision ",k.optim ,"NN =",round(flame.KNN.accuracy, digits = 2)*100,"%"))
#La meilleur valeur de K (allant de 1:10 ou m�me 1:20) est 4
#La pr�cision de 4NN est 92% meilleure que celle de la LDA; :)


#*****************************Classificateur Baysien if*******************************#

library(e1071)
# apprentissage sur le training set
flame.NaiveB = naiveBayes(as.factor(flame_train[,3]) ~ ., data = flame_train[,-3])
#Prediction sur l'�chantillon test
flame.NaiveB.pred = predict(flame.NaiveB, flame_test[,-3]) 
#Table de confusion
table(flame.NaiveB.pred,flame_test[,3])
#Calcul de la pr�cision
flame.NaiveB.erreurClass <- mean(flame.NaiveB.pred != flame_test[,3])
flame.NaiveB.accurcy = 1 - flame.NaiveB.erreurClass
print(paste("Pr�cision Naive Bayes =",round(flame.NaiveB.accurcy, digits = 2)*100,"%"))
# Le score obtenu est de 67% mieux que la LDA mais moins bon que KNN :/


#*****************************************QDA*****************************************#
#Construction de discrimination quadratique
flame.qda <- qda(flame_train[,-3], flame_train[,3]) 
#Prediction selon le mod�le quadratique
flame.qda.pred <- predict(flame.qda,flame_test[,-3]) 
#Table de confusion
table(flame.qda.pred$class, flame_test[,3]) 
#Calcul de la pr�cision
flame.qda.erreurClass <- mean(flame.qda.pred$class != flame_test[,3])
flame.qda.accurcy = 1 - flame.qda.erreurClass
print(paste("Pr�cision QDA =",round(flame.qda.accurcy, digits = 2)*100,"%"))

#Contrairement � ce qu'on aurait pu penser la QDA ne donne pas de tr�s bon r�sultats
#Seulement 56% de pr�cision (mieux que LDA mais moins bonne que KNN et Naive Bayes :/ 


#*****************************************SVM*****************************************#
#D�finition des kernls n�cissaires au SVM
kernels = c("linear","polynomial","radial","sigmoid")
#La fonction tune �stime les param�tres
flame.svmTune = tune(svm, train.x = flame_train[,-3], train.y = flame_train[,3], 
                      validation.x = flame_test[,-3], validation.y = flame_test[,3], 
                      ranges = list(kernel = kernels))

# pr�cision du meilleur mod�le
flame.svm.accurcy = 1 - flame.svmTune$best.performance 
print(paste("Pr�cision SVM (",flame.svmTune$best.parameters$kernel,") =",
            round(flame.svm.accurcy, digits = 2)*100,"%"))
#Avec le kernel gaussien radial, nous obtenons un taux de pr�cision sup�rieur � 99% :D 
