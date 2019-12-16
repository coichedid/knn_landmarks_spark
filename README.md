# Estudos sobre implementação de Knn com Landmarks e Apache Spark
Este repositório traz uma sugestão de implementação do algoritmo KNN (K-Nearest Neighbors) com Landmarks, apresentado por Lima, Gustavo and Mello, Carlos and Silva, Geraldo em 2018 no artigo "A New Modeling for Item Ratings Using Landmarks", utilizando a plataforma Apache Spark.

Este algoritmo é largamente utilizado em sistemas de recomendação e seu treinamento envolve grandes volumes de dados. A execução do produto cartesiano entre a matriz de itens e a de landmarks é o maior desafio da implementação e o artigo traz uma abordagem que utiliza a propagação da matriz de landmarks para todo o cluster Spark.

A implementação envolve a implementação de dois produtos cartesianos na forma de cross join, utilizando o mecanismo de cache do Spark para a distribuição dos dados de uma das matrizes.

O projeto foi desenvolvido com Apache Spark 2.4, pyspark e os pacotes SKLearn e SciPy. Os testes foram executados no cluster spark HDInsights do Microsoft Azure.
