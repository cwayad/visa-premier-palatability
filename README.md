# visa-premier-palatability

---
title: 
author: 
date: 
output:
  pdf_document:
    fig_caption: yes
    keep_tex: yes
    number_sections: yes
  html_document:
    fig_caption: yes
    force_captions: yes
    highlight: pygments
    number_sections: yes
    theme: cerulean
csl: mee.csl
bibliography: references.bib
---

```{r, echo=FALSE}
  # devtools::install_github("cboettig/knitcitations@v1")
  library(knitcitations); cleanbib()
  cite_options(citation_format = "pandoc", check.entries=FALSE)
  library(bibtex)
```
# Introduction {-}
Diff?rentes approches d'apprentissage automatique supervis? existent. Le type de donn?es, leur volume et la disponibilit? sufisante de leur caract?ristiques sont des facteurs centraux du choix des algorithmes performants permettant d'avoir un bon rendu d'apprentissage, lequel est n?cissaire pour l'?tape de pr?diction par la suite.

Dans ce rapport nous allons synth?tiser une ?tude comparative entre plusieurs m?thodes d'apprentissage sur des donn?es synth?tiques dont les labels sont connus. Ensuite, nous appliquons les mod?ls de classement ad?quats sur un cas des lients d'une banque dans le but d'estimer un score d'app?tence ? la carte VISA Premier.

Ce document est constitu? de deux ?tapes majeurs, la premi?re est consacr?e ? l'?tude comparative entre les algorithmes d'apprentissage suppervis? appliqu?s sur trois jeux de donn?es fournis (*flame.r*, *spiral.r* et *aggregation.r*). Quant ? la deuxi?me partie, et apr?s une ?tude exploratoire pr?liminaire de *VisaPremier.txt*, nous discutons l'application des approches que nous jugons convenables aux donn?es r?elles fournises.

#Partie I: Etude sur donn?es synth?tiques {-}
Dans cette partie nous r?alisons une ?tude comparative des diff?rentes approches de classification supervis?e vu en cours telles que :

* Logistic Regression
* Linear discriminant Analysis (LDA)
* Quadratic discriminant Analysis (QDA) 
* Naive Bayes classifier
* K Nearest Neighbours (KNN)
* Support Vector Machine (SVM)

# Analyse :  \emph{Aggregation.txt},  \emph{Flame.txt }et  \emph{Spiral.txt}

```{r echo=FALSE}
# Redirection vers le bon path
setwd("F:/M2-MLDS/Apprentissage supervis?/Projet_app_sup_MLDS16")

#Chargement des donn?es
flame <- read.table("flame.txt",header = T, sep = "\t", fill = T)
spiral <- read.table("spiral.txt",header = T, sep = "\t", fill = T)
aggregation <- read.table("Aggregation.txt", header = T,sep="\t")

```

## \emph{Flame.txt }

Ces donn?es sont repr?sent?es par une matrice de 239 individus d?crits par 2 variables de valeurs quantitatives continues. Ces individus sont divis?s en 2 classes. Il n'y a pas des valeurs manquantes. Les donn?es n'ont pas besoin de m?thodes de pr?traitement et peuvent ?tre utilis?es directement pour l'apprentissage et la pr?diction.

```{r echo=FALSE, fig.height=3, fig.width=6}
#dim(flame)
plot(flame$X1.85, flame$X27.8, col=flame$X1, pch=21,bg=c("blue", "magenta")[as.numeric(flame$X1)], main="Flame")
```

Comme est illustr? dans la figure pr?c?dente, les donn?es forment 2 classes plus au moins s?par?es. Une ligne sous forme d'une parabole peut les bien s?parer. Les algorithme lin?aires ne pourront pas trouver une bonne marge entre les deux classes dans ce cas-l?. 

## \emph{Spiral.txt}

Les 311 individus de cette base sont bivari?s, et sont divis?s en 3 classes. Aucune des m?thodes de pr?traitement n'est n?cissaire ? appliquer sur ces donn?es.

```{r echo=FALSE, fig.height=3, fig.width=6}
#dim(spiral)
plot(spiral$X31.95, spiral$X7.95, col= spiral$X3, pch=21,bg=c("blue", "magenta", "green")[as.numeric(spiral$X3)], main="Spiral")
```

La visualisation des donn?es du fichier *spiral* -comme le montre la figure au dessus- nous permet de distinguer 3 classes spirales bien s?par?es, les classifieurs lin?aires ne pourront pas les distinguer. Cependant, d'autres m?thodes non lin?aires devraient pouvoir trouver ces 3 classes. 

## \emph{Aggregation.txt}
Cette base de donn?es contient 788 individus repr?sent?s par 2 variables dont les valeurs sont quantitatives continues. Nous remarquons qu'il n'y a pas de valeurs manquantes. Les intervalles des valeurs pour les deux variables sont comparables, donc nous n'avons pas besoin de pr?traitement tels que la mise en ?chelle ni la normalisation pour entamer la phase d'apprentissage.
Les ?l?ments de cette base sont regrouper en 7 classes diff?rentes comme est illustr? dans la figure suivante. 

```{r echo=FALSE, fig.height=3, fig.width=6}
#dim(aggregation)
plot(aggregation$X15.55, aggregation$X28.65, col= aggregation$X2, pch=21,bg=c("blue","yellow","red","gray","green", "black", "magenta")[as.numeric(aggregation$X2)], main="Aggregation")
```

Le sch?ma ci-avant montre une bonne s?paration des 7 classes, sauf quelques rares points qui sont ? la limite d'une classe et une autre. Il est clair que les s?parateurs lin?aires (tel que LDA) devraient pouvoir detecter facilement ces 7 classes.

# Comparaison des mod?ls de classification supervis?e

Dans cette section , nous appliquons les diff?rents mod?ls de classement des donn?es sur les 3 ?chantillons fournis. Ensuite, nous synth?tisons une ?tude comaparative entre les performances de chaque mod?l sur chaque base de donn?es dans trois tableaux. Le code utilis? est disponible dans les  fichiers joints *flame.r*, *spiral.r* et *aggregation.r*.

Pour appliquer les diff?rentes approches cit?es auparavant, nous avons consacr? 70% des donn?es de chaque base pour la phase d'apprentissage et 30% pour le test.

## Apprentissage sur \emph{Flame.txt }
Le tableau ci-apr?s r?sume la pr?cision obtenue par les m?thodes cit?es auparavant sur les donn?es *Flame*. Pour le KNN, nous avons utilis? une boucle (de 1 ? 10) pour trouver empiriquement le bon K qui donne par la suite de bonnes performances. Quant au SVM, nous l'avons test? avec plusieurs Kernels ? savoir: *linear*, *polynomial*, *radial* et *sigmoid*
Les pr?cisions trouv?es varient entre celles qui sont bonnes, moyennes et moins bonnes.

M?thode   |Logistic Regression|  LDA  |  QDA  | Naive Bayes |KNN (k=4)|SVM(Radial Guassien)|
---------- ------------------- ------- ------- ----------- ----------- --------------------
Pr?cision |          0.46       0.36    0.56       0.67          0.92          0.99        |

Pour une bonne classification de la base *Flame*, il est recomendable d'utiliser un SVM avec un noyeau radial gaussien car sur un ?chantillon de 100 individus il ne se trompe que dans le classement d'un seul ?l?ment. Le 4NN a aussi un moins d'erruers de classiication. Par contre, LDA, QDA et la regression logistique ne donnent pas de bonnes performances dans l'apprentissage sur les donn?es *Flame*.

## Apprentissage sur \emph{Spiral.txt}

M?thode   |Logistic Regression|  LDA  |  QDA  | Naive Bayes |KNN (k=4)|SVM(Radial Guassien)|
---------- ------------------- ------- ------- ----------- ----------- --------------------
Pr?cision |                

## Apprentissage sur \emph{Aggregation.txt}

M?thode   |Logistic Regression|  LDA  |  QDA  | Naive Bayes |KNN (k=4)|SVM(Radial Guassien)|
---------- ------------------- ------- ------- ----------- ----------- --------------------
Pr?cision |                   

# synth?se et conclusion

L'excustion des m?thodes d'apprentissage cit?es auparavant a donn? des r?sultats diff?rents d'un mod?l ? un autre, cela est du au fait que chaque algorithme a ses points forts et points faibles, car le type, la dispertion et le volume des donn?es sont les facteurs sur lesquels les mod?ls s'appuient.

***
 
#Partie II: Etude d'un cas  pratique {-}

## 1. ?tude exploratoire pr?liminaire {-}
## 2. Classification supervis?e vue en cours pour cr?er un mod?le de scoring.{-}
## 3. variables (disponibles ou uniquement les variables quantitatives et r?aliser ou non une s?lection de variables.){-}
## 4. Comparaison des techniques ? l'aide de {-}
* courbes ROC (AUC), ?valu?es soit par  
+ validation crois?e 
+ ?chantillon test.  

#Annexe

Le code en r est disponible dans le fichier **projet_ASupervise.r** 




