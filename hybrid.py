import pandas as pd
from surprise.dataset import Dataset, Reader
from surprise.model_selection import GridSearchCV
from surprise import SVD
import sqlite3 as sql

from collections import defaultdict
import numpy as np

class SistemaRecomendacionHibrido():

# COLABORATIVO
    # Función que realiza el filtrado colaborativo para recomendar libros a un usuario específico
  def rec_col(self,  usuario):
      # Lee datos de calificaciones desde la base de datos y guarda en un archivo CSV
      con = sql.connect('recomen.db')
      df1 = pd.read_sql_query('SELECT * FROM calificacion', con)
      df1.to_csv("calificacion.csv", index=False)

      # Lee el archivo CSV recién creado y configura el objeto Dataset para Surprise
      df2 = pd.read_csv('calificacion.csv')
      reader = Reader(rating_scale=(1, 5))
      data = Dataset.load_from_df(df2, reader)

      # Configura la cuadrícula de parámetros para la búsqueda de hiperparámetros
      param_grid = {'n_factors': [50, 100, 150], 'n_epochs': [20, 30], 'lr_all': [0.005, 0.01], 'reg_all': [0.02, 0.1]}

      # Realiza la búsqueda de hiperparámetros utilizando validación cruzada
      gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
      gs.fit(data)

      # Obtiene los mejores parámetros encontrados durante la búsqueda
      params = gs.best_params['rmse']

      # Entrena un modelo SVD con los mejores parámetros en el conjunto de entrenamiento completo
      trainset = data.build_full_trainset()
      svdtuned = SVD(**params)
      svdtuned.fit(trainset)

      # Predice las calificaciones para todas las combinaciones (usuario, ítem) que NO están en el conjunto de entrenamiento
      testset = trainset.build_anti_testset()
      predictions = svdtuned.test(testset)

      # Obtiene las mejores predicciones para cada usuario (top 3)
      top_n = self.get_top_n(predictions, n=3)
      con.close()
      # Retorna la lista de recomendaciones colaborativas para el usuario dado
      return self.recocolaborativa(usuario, top_n)

  # Función para obtener las mejores predicciones para cada usuario
  def get_top_n(self, predictions, n=10):
      top_n = defaultdict(list)
      for uid, iid, true_r, est, _ in predictions:
          top_n[uid].append((iid, est))
      for uid, user_ratings in top_n.items():
          user_ratings.sort(key=lambda x: x[1], reverse=True)
          top_n[uid] = user_ratings[:n]
      return top_n

  # Función para obtener recomendaciones colaborativas
  def recocolaborativa(self, usuario, top_n):
      recoco = []
      for pelicula, calificacion in top_n[usuario]:
          recoco.append([pelicula, calificacion])
      return recoco

##############################################################

# CONTENIDO

  def rec_cont(self,usuario):
    # Lee datos de libros desde la base de datos
    consulta = 'select * from libros'
    con = sql.connect('recomen.db')
    libros = pd.read_sql_query(consulta, con)

    # Transforma las columnas de la categoría "Publicador" en variables dummy
    libros_categ = pd.get_dummies(libros, prefix_sep='', prefix='', columns=['autor'])

    # Elimina columnas no necesarias para el filtrado por contenido
    libros_categ = libros_categ.drop(["editorial", "genero", "link1","link2", "comments", "estado"], axis=1)

    # Guarda los datos transformados en una nueva tabla SQLite
    libros_categ.to_sql("recomendador_contenido", con, if_exists="replace", index=False)

    # Obtiene las calificaciones del usuario desde la base de datos
    query = 'SELECT titulo, calificacion FROM calificacion WHERE codigo = "' + usuario + '"'
    calificacion = pd.read_sql_query(query, con)

    # Inicializa un DataFrame para almacenar los libros calificados
    calificados = pd.DataFrame()

    # Obtiene la información de los libros calificados por el usuario
    for i in range(0, len(calificacion)):
        query = 'SELECT * FROM recomendador_contenido WHERE titulo = "' + calificacion.loc[i][0] + '" '
        consulta = pd.read_sql_query(query, con)
        # calificados = calificados.add(consulta)
        calificados = pd.concat([calificados, consulta], ignore_index=True)
    # Obtiene el rango de peso_categoria de los libros calificados
    rango_calificados = [min(list(calificados['peso_categoria'])), max(list(calificados['peso_categoria']))]

    # Combina las calificaciones con la información de los libros
    libros2 = pd.merge(calificacion, calificados, on='titulo', suffixes=('_x', '_y'))

    # Calcula el perfil del usuario sin normalizar
    for i in range(len(libros2)):
        for j in range(2, libros2.loc[0].count()):
            libros2.iat[i, j] = libros2.iat[i, 1] * libros2.iat[i, j]

    perfil_sin_normalizar = libros2.sum(axis=0)

    # Calcula las sumas totales para normalizar
    total_gamemode = perfil_sin_normalizar[1] + perfil_sin_normalizar[2]
    total_nota = perfil_sin_normalizar['calificacion']
    total_publisher = 0

    for i in range(5, len(perfil_sin_normalizar)):
        total_publisher = total_publisher + perfil_sin_normalizar[i]

    # Inicializa un nuevo DataFrame para el perfil normalizado
    perfil_normalizado = libros2.astype('float64', errors='ignore')

    # Normalización de datos
    for i in range(1, len(libros2)):
        perfil_normalizado = perfil_normalizado.drop(i)

    for i in range(2, perfil_normalizado.loc[0].count()):
        if i <= 3:
            perfil_normalizado.iat[0, i] = perfil_sin_normalizar[i] / total_gamemode
        elif i == 4:
            perfil_normalizado.iat[0, i] = perfil_sin_normalizar[i] / total_nota
        else:
            perfil_normalizado.iat[0, i] = perfil_sin_normalizar[i] / total_publisher

    perfil_normalizado = perfil_normalizado.drop(["titulo", "calificacion"], axis=1)

    ## Se normaliza el peso_categoria para el perfil
    prn = perfil_normalizado['peso_categoria'][0]
    prn = (prn - rango_calificados[0]) / (rango_calificados[1] - rango_calificados[0])
    perfil_normalizado['peso_categoria'][0] = prn

    libros_categ2 = libros_categ

    # Elimina los juegos calificados del conjunto de libros
    for i in range(len(libros2)):
        libros_categ2 = libros_categ2.drop(libros_categ[libros_categ['titulo'] == libros2.loc[i][0]].index)

    # Normalización del peso_categoria para los libros no calificados
    rango_peso_categorias = [min(list(libros_categ2['peso_categoria'])), max(list(libros_categ2['peso_categoria']))]

    for i in range(len(libros_categ2)):
        prn = libros_categ2['peso_categoria'].iat[i]
        prn = (prn - rango_peso_categorias[0]) / (rango_peso_categorias[1] - rango_peso_categorias[0])
        libros_categ2['peso_categoria'].iat[i] = prn

    # Calcula la similitud coseno de cada juego
    sim = []
    for i in range(len(libros_categ2)):
        libro = list(libros_categ2.iloc[i][1:])
        usuario = list(perfil_normalizado.iloc[0])
        sim += [np.dot(libro, usuario) / (np.linalg.norm(libro) * np.linalg.norm(usuario))]

    # Agrega la similitud coseno como una nueva columna en el DataFrame de juegos no calificados
    libros_categ2['Similitud'] = sim

    sim = np.array(sim)
    # Se lleva similitud coseno a calificacion
    sim = (sim * 4) + 1

    nombres = list(libros_categ2['titulo'])

    listCon = []

    # Crea la lista de recomendaciones por contenido ordenada por similitud descendente
    for n in range(len(nombres)):
        listCon += [[nombres[n], sim[n]]]
    con.close()
    return sorted(listCon, key=lambda x: x[1], reverse=True)


##############################################################

# HIBRIDO

  def get_recomendations(self,usuario):
    pesoContenido = 0.4
    pesoColaborativo = 0.6

    listaCon = self.rec_cont(usuario)
    listaCol = self.rec_col(usuario)
    rangolcl = len(listaCol)
    rangolcn = len(listaCon)
    items_peso = {}
    for i in range(rangolcl):
        try:
            nuevoCol= listaCol[i][1] * pesoColaborativo
            items_peso[listaCol[i][0]] = max(items_peso[listaCol[i][0]],nuevoCol)
        except KeyError:
            items_peso[listaCol[i][0]] = listaCol[i][1] * pesoColaborativo

    for i in range(rangolcn):
        try :
            nuevoCon = listaCon[i][1]* pesoContenido
            items_peso[listaCon[i][0]] = max(items_peso[listaCon[i][0]],nuevoCon)
        except KeyError:
            items_peso[listaCon[i][0]] = listaCon[i][1] * pesoContenido

    libros_rec = sorted(items_peso.items(),  key=lambda x: x[1], reverse = True)

    return libros_rec






