import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql import Row
from functools import reduce, partial
from operator import add
from scipy.spatial import distance as dis
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import minmax_scale
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import Window
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

def save_cross_join(X, Y, base_hdfs, name):
    """Cria um dataframe com o produto cartesiano de duas matrizes.

    :param DataFrame X: Matriz 1.
    :param DataFrame Y: Matriz 2.
    :param str base_hdfs: Caminho base para salvar o novo dataframe.
    :param str name: Nome do arquivo parquet.
    :return:
    :rtype: None

    """
    X_columns = ['x_{}'.format(c) for c in X.columns]
    Y_columns = ['y_{}'.format(c) for c in Y.columns]
    X_ren_ = X.toDF(*X_columns)
    X_ren = X_ren_.repartition(12)
    Y_ren_ = Y.toDF(*Y_columns)
    Y_ren = Y_ren_.repartition(12).cache()
    Y_ren.take(1)
    spark.sql('set spark.sql.autoBroadcastJoinThreshold=0')

    cross = X_ren.join(Y_ren)
    cross.repartition(12).write \
        .mode('append') \
        .parquet('{}{}.parquet'.format(base_hdfs, name))

def get_distance(X_features, Y_features, metric):
    """Calcula a distancia entre duas listas com a métrica fornecida.

    :param list X_features: Lista 1.
    :param list Y_features: Lista 2.
    :param func metric: Função de cálculo da distância.
    :return: Valor da distância.
    :rtype: DoubleType ou None

    """
    if (X_features is not None) and (Y_features is not None) and (metric is not None):
        distance = metric(X_features, Y_features)
        return distance.item()

def to_null(c):
    """Completa uma feature com nulo caso ela não esteja preenchida com número ou string.

    :param string c: Nome da coluna.
    :return: Null ou valor da coluna.
    :rtype: type

    """
    return f.when(~(f.col(c).isNull() | f.isnan(f.col(c)) | (f.trim(f.col(c)) == "")), f.col(c))

def get_neighbors(row, k_nn):
    """Para cada linha, busca os vizinhos mais próximos.

    :param Row row: Uma linha de usuário.
    :param int k_nn: Quantidade de vizinhos.
    :param double lim_similarity: Limite de similaridade.
    :return: Linha [item, num_vizinhos, vizinhos[]]
    :rtype: Row

    """

    d = row.asDict()
    user = d['user']
    other_users = [d[k] for d.keys() if k != 'user']
    # other_users = [(i, other_users[i]) for i in np.arange(len(other_users))]
    neighbors = sorted(other_users, reverse = True)
    neighbors = neighbors[1:k_nn+1]

    new_row = {}
    new_row['user'] = user
    new_row['count'] = len(neighbors)
    new_row['neighbors'] = neighbors
    return Row(**new_row)

def scale(row, prefix, label_1, label_2, label_3):
    """Escala os dados de uma coluna conforme os mínimos e máximos.

    :param Row row: Linha a ser escalada.
    :return: Nova linha.
    :rtype: Row

    """

    d = row.asDict()
    min = d['min']
    max = d['max']
    y_label = d['{}.{}'.format(prefix, label_2)]
    x_label = label_1
    v_x_label = d[x_label]
    v_y_label = d[y_label]
    v_value = d[label_3]
    new_row = {}
    new_row[x_label] = v_x_label
    new_row[y_label] = v_y_label

    v_std = (v_value - min) / (max - min)
    new_row[label_3] = v_std
    return Row(**new_row)

def map_fn(row):
    """Mapeia os valores faltantes para a média.

    :param Row row: Linha da matriz.
    :return: Nova linha com as médias preenchidas.
    :rtype: Row

    """
    new_row_dict = {}
    d = row.asDict()
    mean = d['mean']
    user = d['user']
    for (c, v) in d.items():
        if c != 'mean' and c != 'user':
            new_v = int(v) if v is not None else int(mean)
            new_row_dict[c] = new_v
    new_row_dict['user'] = user
    new_row_dict['mean'] = mean
    return Row(**new_row_dict)

# Parametriza a função de distância para ser executada com spark
metric = dis.correlation
partial_get_distance = partial(get_distance, metric=metric)
get_distance_udf = f.udf(partial_get_distance, t.DoubleType())

unlist = f.udf(lambda x: round(float(list(x)[0]),3), t.DoubleType())

# Recupera o contexto spark
conf = SparkConf().setAppName(args['JOB_NAME']).set('spark.executor.memoryOverhead', 1000)
sc = SparkContext.getOrCreate(conf)
spark = SparkSession.builder.getOrCreate()


files = [
    ('ratings', 'u-1m.data', ['src','dst','Rating','Timestamp']),
    ('movies', 'u-1m.item', ['id', 'Title', 'Genres']),
    ('users', 'users.dat', ['id','Gender','Age','Occupation','Zipcode'])
]

## Parâmetros
sp_dfs = {}
master_host = 'hn0-dbiisp'
base_hdfs = 'hdfs://{}/'.format(master_host)
n_knn = 100
n_landmarks = 250
landmark_feature = 'user'
measured_feature = 'movie'
measure_feature = 'rating'
k_fold = [.8, .2]

## Recuperando dados
for name, f, columns in files:
    df = spark.read.parquet('{}{}.parquet'.format(base_hdfs, name))
    sp_dfs[name] = df
    df.registerTempTable(name)

ratings = spark.sql('select src as user, dst as movie, Rating as rating from ratings')
users = sp_dfs['users']
movies = spark.sql('select id, Title as title from movies')

ratings.registerTempTable('tb_ratings')
users.registerTempTable('tb_users')
movies.registerTempTable('tb_movies')

movie_list = [str(getattr(r, 'id')) for r in movies.select('id').distinct().collect()]
movie_list.sort()


# Separando o dataset em treino e teste
train_data = spark.read.parquet('{}{}.parquet'.format(base_hdfs, 'train_data'))
test_data = spark.read.parquet('{}{}.parquet'.format(base_hdfs, 'test_data'))

# Selecionando os landmarks
landmarks = train_data.groupBy(landmark_feature)\
    .count()\
    .sort(f.col("count").desc())\
    .select(landmark_feature)\
    .limit(n_landmarks)
landmarks = landmarks.alias('landmarks')
landmarks.repartition(12).write \
        .mode('append') \
        .parquet('{}{}.parquet'.format(base_hdfs, 'landmarks_selected'))

landmarks = spark.read.parquet('{}{}.parquet'.format(base_hdfs, 'landmarks_selected'))
landmarks = landmarks.alias('landmarks')
landmarks_ids = [str(getattr(r, landmark_feature)) for r in landmarks.collect()]

landmark_feature_means = train_data.groupBy(landmark_feature)\
    .mean()\
    .select([f.col(landmark_feature), f.col('avg({})'.format(measure_feature)).alias(measure_feature)])\
    .withColumn('round', f.round(f.col(measure_feature), 0).cast('integer'))\
    .select([f.col(landmark_feature), f.col('round').alias(measure_feature)]).alias('means')

# Preparando a matriz de itens
matrix = train_data.groupBy(landmark_feature)\
        .pivot(measured_feature, movie_list)\
        .agg(f.max(measure_feature)).alias('matrix')

matrix_mean = matrix.join(landmark_feature_means, f.col('matrix.{}'.format(landmark_feature)) == f.col('means.{}'.format(landmark_feature)))\
                    .select([f.col('matrix.*'), f.col('means.{}'.format(measure_feature)).alias('mean')]).alias('matrix')


# Completando a matriz com a média
columns = matrix_mean.columns
matrix_completed = matrix_mean.rdd.map(map_fn).toDF().drop('mean')
matrix_completed = matrix_completed.alias('matrix')

# Criando a matriz de landmarks
landmark_data = matrix_completed.join(landmarks, f.col('matrix.{}'.format(landmark_feature)) == f.col('landmarks.{}'.format(landmark_feature)))\
                    .select([f.col('matrix.*')]).alias('landmark_matrix')

name = 'landmark_data'
landmark_data.repartition(12).write \
        .mode('append') \
        .parquet('{}{}.parquet'.format(base_hdfs, name))
landmark_data = spark.read.parquet('{}{}.parquet'.format(base_hdfs, name))
name = 'matrix_completed'
matrix_completed.repartition(12).write \
        .mode('append') \
        .parquet('{}{}.parquet'.format(base_hdfs, name))
matrix_completed = spark.read.parquet('{}{}.parquet'.format(base_hdfs, name))

# Cria a matriz do produto cartesiano
name = 'cross_matrix'
save_cross_join(matrix_completed, landmark_data, base_hdfs, name)
cross = spark.read.parquet('{}{}.parquet'.format(base_hdfs, name))

# Renomeia as colunas para separar as features de itens e de landmarks
X_col_names = ['x_{}'.format(c) for c in matrix_completed.columns if c != 'user']
X_cols = [f.col(c) for c in X_col_names]
X_columns = ','.join(X_col_names)
Y_col_names = ['y_{}'.format(c) for c in landmark_data.columns if c != 'user']
Y_cols = [f.col(c) for c in Y_col_names]
Y_columns = ','.join(Y_col_names)

# Calcula as similaridades (distâncias)
cross2 = cross.withColumn('x_features', f.array(X_cols))
cross2 = cross2.withColumn('y_features', f.array(Y_cols))
similarities = cross2.select(f.col('x_user').alias('user'), f.col('y_user').alias('y_label'), get_distance_udf('x_features', 'y_features').alias('distance'))

# Elimina linhas em branco
similarities_a = similarities.select([to_null(c).alias(c) for c in similarities.columns]).na.drop()
name = 'distancias'
similarities_a.repartition(12).write \
        .mode('append') \
        .parquet('{}{}.parquet'.format(base_hdfs, name))
similarities_a = similarities_a.alias('s1')

# Recupera mínimos e máximos para normalizar a matriz de similaridades
s_1 = similarities.groupBy('y_label').agg(f.min('distance'), f.max('distance')).show(2)
s_1 = s_1.alias('s2')

similarities_joined = similarities_a.join(s_1, f.col('s1.y_label') == f.col('s2.y_label'))
# Normalizando a matrix
partial_scale = partial(scale, prefix = 's1',
        label_1 = 'user',
        label_2 = 'y_label',
        label_3 = 'distance')
similarities_scaled = similarities_joined.rdd.map(partial_scale).toDF()

# Criando a matriz de usuários por landmarks
# Esta matriz é a projeção dos usuários no espaço de landmarks
similarities2 = similarities_scaled.groupBy('user')\
                .pivot('y_label', landmarks_ids)\
                .agg(f.max('distance'))

name = 'similaridades'
similarities2.repartition(4).write \
        .mode('append') \
        .parquet('{}{}.parquet'.format(base_hdfs, name))
similarities_landmarks = spark.read.parquet('{}{}.parquet'.format(base_hdfs, name))

# Agora é necessário computar a similaridade entre usuários e usuários
# Como todos os usuários estão projetados no plano de landmarks, as features
# foram normalizadas e tem-se uma matriz completa.
# O processo é similar ao processo de projeção dos landmarks

# Cria a matriz do produto cartesiano Usuário X Usuário
name = 'cross_matrix_u_u'
save_cross_join(similarities_landmarks, similarities_landmarks, base_hdfs, name)
cross_u_u = spark.read.parquet('{}{}.parquet'.format(base_hdfs, name))

# Renomeia as colunas para separar as features de itens e de landmarks
X_col_names = ['x_{}'.format(c) for c in similarities_landmarks.columns if c != 'user']
X_cols = [f.col(c) for c in X_col_names]
X_columns = ','.join(X_col_names)
Y_col_names = ['y_{}'.format(c) for c in similarities_landmarks.columns if c != 'user']
Y_cols = [f.col(c) for c in Y_col_names]
Y_columns = ','.join(Y_col_names)

# Calcula as similaridades (distâncias) Usuário X Usuário
cross2_u_u = cross.withColumn('x_features', f.array(X_cols))
cross2_u_u = cross2_u_u.withColumn('y_features', f.array(Y_cols))
similarities_u_u = cross2_u_u.select(f.col('x_user').alias('user'), f.col('y_user').alias('y_label'), get_distance_udf('x_features', 'y_features').alias('distance'))

# Encontra os n vizinhos próximos
partial_get_neighbors = partial(get_neighbors, k_nn = n_knn)
neighbors = similarities_u_u.rdd.map(partial_get_neighbors).toDF()

## O próximo passo é realizar a estimativa do dataset de teste [usuario, filme]
### Notas do Usuário matrix_completed[usuário]
### Média das Notas do Usuário matrix_completed[usuario].mean()
### Notas do Filme matrix_completed[:filme]
### Vizinhos do usuário neighbors[usuario]
### Valores das notas dos vizinhos
### Médias das notas dos vizinhos
##### para cada vizinho
########    matrix_completed[vizinho]
########    matrix_completed[vizinho].mean()

# Para otimizar esses passos, os dataframes matrix_completed e neighbors
# devem ser transformadas em variaveis locais e distribuídas nos nos do cluster
# Dessa forma, é possivel percorrer o DataFrame de treino estimando os valores
# test_data.select['user', 'movie', partial_estimate_value(f.col('user'), f.col('movie'))]
