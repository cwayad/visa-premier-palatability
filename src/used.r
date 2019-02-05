# Redirection vers le bon path
setwd("./data")


########## Donn�es flame ##########
#Chargement des donn�es
flame = read.table('data/flame.txt')

#On scinde la table en deux parties par les colonnes variables et classes
flame_var = flame[,1:2] #Les deux variables
flame_cla = flame[,3] #Les classes (2 classes)

#On plot les variables et on colries en fonction des classes
plot(flame_var,col=flame_cla)
#La on voit bien les deux classes s�par�es par une courbe parabolique

#Il va falloir bien s�parer les ensembles de test et de train
#On va prendre 20 individus comme �chantillon de test 
#� moiti� de la classe 1 et moiti� de la classe 2
test_flame_var = flame_var[144:163,]
train_flame_var = rbind(flame_var[1:143,],flame_var[164:240,])

test_flame_cla = flame_cla[144:163]
train_flame_cla = c(flame_cla[1:143],flame_cla[164:240])

#La pr�cision semble �tre une bonne mesure pour �sitmer les m�thodes sur ce jeu de donn�es
#Le nombre d'individu par classe est assez similaire

#LDA
library(MASS)
flame_z_lda <- lda(train_flame_var, train_flame_cla) #Construction de discrimination lin�aire
flame_zp_lda <- predict(flame_z_lda,test_flame_var) #Prediction selon le mod�le lin�aire
#Table de confusion
table(test_flame_cla,flame_zp_lda$class) #Nous montre une pr�cision tr�s basse de 25%
#Pr�cision que l'on calcule
accurcy_lda_flame = 1 - ( sum(flame_zp_lda$class != test_flame_cla) / length(test_flame_cla) )

#KNN
library(class)
# Ici le bon K n'est pas pr� d�termin�
# On va boucler sur diff�rents K de 1 � 10 et on sauvegarde la meilleure performance
best_k = 0
best_KNN_acc_flame = 0
for (i in 1:10) {
  flame_knn = knn(train_flame_var, test_flame_var, cl = train_flame_cla, k = i)
  tmp_acc = 1 - ( sum(flame_knn != test_flame_cla) / length(test_flame_cla) )
  if(tmp_acc > best_KNN_acc_flame){
    best_KNN_acc_flame = tmp_acc
    best_k = i
  }
}
#Le meilleur K obtenu est 4 et avec 4NN on obtien une pr�cision de 0.9 d�j� meilleur que la LDA

#Classificateur baysien naif
library(e1071)
#Cr�ation du mod�le
MBN_flame = naiveBayes(as.factor(train_flame_cla) ~ ., data = train_flame_var)
flame_zp_MBN = predict(MBN_flame, test_flame_var) #Pr�diction � l'aide du mod�le pour le tester
accurcy_MBN_flame = 1 - ( sum(flame_zp_MBN != test_flame_cla) / length(test_flame_cla) )
#Et la on obtien un score de 55% mieux que la LDA mais bon pire que KNN

#QDA
flame_z_qda <- qda(train_flame_var, train_flame_cla) #Construction de discrimination quadratique
flame_zp_qda <- predict(flame_z_qda,test_flame_var) #Prediction selon le mod�le quadratique
#Table de confusion
table(test_flame_cla,flame_zp_qda$class) #Nous montre une pr�cision moyenne
#Pr�cision que l'on calcule
accurcy_qda_flame = 1 - ( sum(flame_zp_qda$class != test_flame_cla) / length(test_flame_cla) )
#Contrairement � ce qu'on aurait pu penser la QDA ne donne pas de tr�s bon r�sultats
#Seulement 55% de pr�cision (� �galit� avec le NBM)

#SVM
#M�thodes g�niale, mais couteuse, et probl�me de choix de kernel donc pas de magie
SVM_kernels = c("linear","polynomial","radial","sigmoid")
#On utilise la fonction tune qui nous �stime les param�tre, pas besoin de trop coder (Merci monsieur Labiod)
flame_tune_svm = tune(svm, train.x = train_flame_var, train.y = train_flame_cla, validation.x = test_flame_var, validation.y = test_flame_cla, ranges = list(kernel = SVM_kernels, cost = c(0.001, 0.01, 0.1, 1,5,10,100)))
1 - flame_tune_svm$best.performance # pr�cision du meilleur mod�le
flame_tune_svm$best.parameters #avec les meilleurs param�tres
#On obient un taux de pr�cision sup�rieur � 99.5% ! (Kernel gaussien radial, cout = 100)

########## donn�es spiral ##########
#Lecture des donn�es spiral
spiral = read.table('data/spiral.txt')

#S�paration de la table en variables et classes
spiral_var = spiral[,1:2]
spiral_cla = spiral[,3] #Il y a trois classes

#Dessin d'un nuage de points pour visualiser la structure de donn�es
plot(spiral_var, col = spiral_cla)
#La forme bien que jolie des donn�es n'est pas un cadeau pour les m�thodes lin�aires

#S�paration en donn�es de test et donn�es de train
#Pour prendre des �l�ments random on prend la matrice de base et on permute les lignes al�atoirement
random_spiral = spiral[sample(nrow(spiral)),]
#Puis on s�pare ces donn�es m�lang�s en classes et variables
spiral_var = random_spiral[,1:2]
spiral_cla = random_spiral[,3]

#Puis finalement on s�parre la partie test de la partie train
train_spiral_var = spiral_var[1:280,] #Les premier 280 individus pour le train
train_spiral_cla = spiral_cla[1:280]
test_spiral_var  = spiral_var[280:312,] #les 32 autres indivdus pour le test
test_spiral_cla  = spiral_cla[280:312]

#LDA
#Testons la LDA m�me si la s�paration lin�aire sur des donn�es comme celles-ci ne vaut rien
spiral_z_lda <- lda(train_spiral_var, train_spiral_cla) #Construction de discrimination lin�aire
spiral_zp_lda <- predict(spiral_z_lda,test_spiral_var) #Prediction selon le mod�le lin�aire
#Table de confusion
table(test_spiral_cla,spiral_zp_lda$class) #Nous montre une pr�cision basse
#Pr�cision que l'on calcule
accurcy_lda_spiral = 1 - ( sum(spiral_zp_lda$class != test_spiral_cla) / length(test_spiral_cla) )
#On obtien donc 1/3 de pr�cision, �a me surprend qu'il puisse trouver mieux que sur les donn�s flame

#KNN
#La encore pas de K pr� d�fini donc boucle de 1 � 10
best_k_spiral = 0
best_KNN_acc_spiral = 0
for (i in 1:10) {
  spiral_knn = knn(train_spiral_var, test_spiral_var, cl = train_spiral_cla, k = i)
  tmp_acc = 1 - ( sum(spiral_knn != test_spiral_cla) / length(test_spiral_cla) )
  if(tmp_acc > best_KNN_acc_spiral){
    best_KNN_acc_spiral = tmp_acc
    best_k_spiral = i
  }
}
#KNN avec un seul voisin (k=1) s'en sort parfaitement et donne un pr�cision de 1 !
#On voit bien sur le scatter plot des donn�es que le plus proche �l�ment
#a de tr�s forte chances d'appartenir � la m�me classe

##Classificateur baysien naif
MBN_spiral = naiveBayes(as.factor(train_spiral_cla) ~ ., data = train_spiral_var)
spiral_zp_MBN = predict(MBN_spiral, test_spiral_var) #Pr�diction � l'aide du mod�le pour le tester
accurcy_MBN_spiral = 1 - ( sum(spiral_zp_MBN != test_spiral_cla) / length(test_spiral_cla) )
#Le classificateur baysien naif n'est m�me pas aussi bon que la LDA et a 36,36% de pr�cision

#QDA
spiral_z_qda <- qda(train_spiral_var, train_spiral_cla) #Construction de discrimination quadratique
spiral_zp_qda <- predict(spiral_z_qda,test_spiral_var) #Prediction selon le mod�le quadratique
#Table de confusion
table(test_spiral_cla,spiral_zp_qda$class) #Nous montre une pr�cision basse
#Pr�cision que l'on calcule
accurcy_qda_spiral = 1 - ( sum(spiral_zp_qda$class != test_spiral_cla) / length(test_spiral_cla) )
#Seulement 33% des donn�es de test sont bien class�es en se basant sur le mod�le quadratique pour
#ce type des donn�es, et on s'y attendais un peu (moins bien que le mod�le baysien) 

#SVM
#Encore une fois pour estimer les param�tre (meilleur kernel,meilleur cout,r�sultats du meilleur mod�le)
#On utilise la fonction tune
spiral_tune_svm = tune(svm, train.x = train_spiral_var, train.y = train_spiral_cla, validation.x = test_spiral_var, validation.y = test_spiral_cla, ranges = list(kernel = SVM_kernels, cost = c(0.001, 0.01, 0.1, 1,5,10,100)))
1 - spiral_tune_svm$best.performance # pr�cision du meilleur mod�le
spiral_tune_svm$best.parameters
#La on une pr�cision de plus de 98,5% ! (Kernel gaussien radial, cout = 100)

########## donn�es Aggregation ##########
#Lecture de donn�es
aggregation = read.table('data/Aggregation.txt')

#Un scatter plot la encore pour voir les donn�es
plot(aggregation[,1:2],col = aggregation[,3])
#Les classes sont �parpill�es sur le plot, certaines sont bien s�par�es d'autres non
#Certaines sont plus grandes que d'autres, on va bien voir

#S�paration d'un ensemble de test et de train
#Les classes 6 et 7 ne contiennent pas beaucoup d'individus (34 chacune)
#Par peur de les vider plusieur essais de random sont fait
random_aggregation = aggregation[sample(nrow(aggregation)),]
#S�paration en variables et classes
aggregation_var = random_aggregation[,1:2]
aggregation_cla = random_aggregation[,3] #7classes
#S�paration en ensemble de test et de train
train_aggregation_var = aggregation_var[1:688,]
train_aggregation_cla = aggregation_cla[1:688]
test_aggregation_var = aggregation_var[689:788,]
test_aggregation_cla = aggregation_cla[689:788]

#LDA
aggregation_z_lda <- lda(train_aggregation_var, train_aggregation_cla) #Construction de discrimination lin�aire
aggregation_zp_lda <- predict(aggregation_z_lda,test_aggregation_var) #Prediction selon le mod�le lin�aire
#Table de confusion
table(test_aggregation_cla,aggregation_zp_lda$class) #Nous montre une pr�cision tr�s haute
#Pr�cision que l'on calcule
accurcy_lda_aggregation = 1 - ( sum(aggregation_zp_lda$class != test_aggregation_cla) / length(test_aggregation_cla) )
#Et on a un score magnifique de 98% !

#KNN
#Boucle sur le K again
best_k_aggregation = 0
best_KNN_acc_aggregation = 0
for (i in 1:10) {
  aggregation_knn = knn(train_aggregation_var, test_aggregation_var, cl = train_aggregation_cla, k = i)
  tmp_acc = 1 - ( sum(aggregation_knn != test_aggregation_cla) / length(test_aggregation_cla) )
  if(tmp_acc > best_KNN_acc_aggregation){
    best_KNN_acc_aggregation = tmp_acc
    best_k_aggregation = i
  }
}
#La encore KNN avec un seul voisin est excellent, et a un score de pr�cision de 1
#Mais je pense que l'ensemble de test ne contient pas d'�l�ments situ�s dans les fronti�res
#des classes qui se chevauchent un peu

#Classificateur Baysien naif
MBN_aggregation = naiveBayes(as.factor(train_aggregation_cla) ~ ., data = train_aggregation_var)
aggregation_zp_MBN = predict(MBN_aggregation, test_aggregation_var) #Pr�diction � l'aide du mod�le pour le tester
accurcy_MBN_aggregation = 1 - ( sum(aggregation_zp_MBN != test_aggregation_cla) / length(test_aggregation_cla) )
#Le classificateur baysien est tout aussi bon que la LDA sur ces donn�es : 98%

#QDA
aggregation_z_qda <- qda(train_aggregation_var, train_aggregation_cla) #Construction de discrimination quadratique
aggregation_zp_qda <- predict(aggregation_z_qda,test_aggregation_var) #Prediction selon le mod�le quadratique
#Table de confusion
table(test_aggregation_cla,aggregation_zp_qda$class) #Nous montre une pr�cision tr�s haute
#Pr�cision que l'on calcule
accurcy_qda_aggregation = 1 - ( sum(aggregation_zp_qda$class != test_aggregation_cla) / length(test_aggregation_cla) )
#Pr�cision de 99% seul un �l�ment a �t� mal class� par le mod�le quadratique !

#SVM
#Cela va prendre un peu de temps 
aggregation_tune_svm = tune(svm, train.x = train_aggregation_var, train.y = train_aggregation_cla, validation.x = test_aggregation_var, validation.y = test_aggregation_cla, ranges = list(kernel = SVM_kernels, cost = c(0.001, 0.01, 0.1, 1,5,10,100)))
1 - aggregation_tune_svm$best.performance # pr�cision du meilleur mod�le
aggregation_tune_svm$best.parameters
