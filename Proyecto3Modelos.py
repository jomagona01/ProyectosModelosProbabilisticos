'''
Proyecto 3 Modelos Probabilísticos
Jose Mario Gonzalez Abarca - B83362
10/20/2022
'''

import requests
import pandas as pd
from pandas import json_normalize
import matplotlib.pyplot as plt
from scipy import stats
from fitter import Fitter
from scipy.stats import expon, skew, kurtosis, exponweib, burr12, norm
import numpy as np
import statistics 
import random

#Se definen las fechas iniciales y finales para hacer el análisis
#Se va a utilizar el periodo de tiempo de todo el 2019
inicio1='20190101' 
fin1='20191231'


'''
Función 1, Recibe como parámetros la fecha inicial y la fecha final para realizar el análisis.
Extrae mediante request los datos desde el servidor del ICE que contiene toda la información.
Convierte en Json y finalmente en data frame los datos para facilitar su uso en los posteriores
puntos.
'''
#Se crea la función para obtener los datos en un intervalo entre inicio y fin
#Se pasan las fechas de inicio y fin como parámetros
#Se convierte la información extraída del servidor, en un JSon de python
#Una vez con el JSon se convierte este en un data frame
#Se separa la columna de fechaHora, en dos, fecha y hora
#Se concatenan estas dos nuevas columnas con el data frame original, y se elimina la columna fechaHora
#Se retorna el nuevo data frame con los datos separados
def datos_demanda(inicio, fin):

    # Construir la URL
    api_url = 'https://apps.grupoice.com/CenceWeb/data/sen/json/DemandaMW?inicio='+ inicio + '&fin=' + fin 

    # Hacer la solicitud GET y guardar un "Response" en la variable r
    r = requests.get(api_url)

    # Convertir la información obtenida en JSON
    datos = r.json()

    #Primero se convierte el json a data frame
    df = json_normalize(datos['data']) 
    df2 = df["fechaHora"].str.split(expand=True) #Se separan la columna en fecha y hora
    df2.columns = ['Fecha', 'Hora'] #Se nombran las dos nuevas columnas
    dff= pd.concat([df, df2], axis=1) #Se combina el nuevo df de las columnas separadas con el original
    df_ordenado=dff.drop(columns = ["fechaHora"]) #Se elimina la columna con la fecha y hora combinadas
    
    return df_ordenado

#Data frame de los datos a utilizar en todo el proyecto (datox)
datox = datos_demanda(inicio1, fin1)




##############################################
#PUNTO 2

'''
Función 2, recibe como parámetros el data frame con la información delimitada,
y la hora que se quiere analizar.
Devuelve un data frame únicamente con todas las potencias que se tienen a esa hora.
'''
#Para este punto inicialmente se obtienen las horas asignadas
def asignacion_horas(digitos):
    '''Elige una hora A en periodo punta
    y una hora B de los otros periodos,
    con los dígitos del carné como "seed"
    '''
    
    random.seed(digitos)
    punta = [11, 12, 18, 19, 20]
    valle = [7, 8, 9, 10, 13, 14, 15, 16, 17]
    nocturno = [21, 22, 23, 0, 1, 2, 3, 4, 5, 6]
    otro = valle + nocturno
    hora_A = random.choice(punta)
    hora_B = random.choice(otro)
    return hora_A, hora_B

horax, horay = asignacion_horas(83362)
#Las horas asignadas fueron 11 y 15 --- 11am y 3pm

#Se pasan las horas a string que reconozca el data frame
hora1 = "11:00:00.0"
hora2 = "15:00:00.0"

#Se crea la función que obtiene los datos de potencia de una hora específica
#Se pasa como parámetro el data frame y la hora específica
#Se retorna la potencia en MWh de esa hora
def datos_hora(df, hora):
    df2=df[df['Hora']==hora]

    #Se guardan todos los diametros de los arboles seleccionados en la lista anterior
    df3=df2['MW']
    
    return df3

consumo_hora1=datos_hora(datox, hora1)
consumo_hora2=datos_hora(datox, hora2)


###############################################
#PUNTO 3

'''
Función 3, recibe como parámetros el data frame delimitado, y la hora de análisis.
Llama a la función anterior para agrupar los datos y finalmente obtiene el modelo 
de mejor ajuste de estos.
'''

#Inicialmente se llama a la función datos-hora creada anteriormente para guardar el consumo de esa hora específica
#Una vez teniendo los datos de consumo se obtienen los parámetros por medio de Fitter
#Finalmente se grafica el histograma con el mejor modelo de ajuste y algunos de lo mas aproximados
def modelo_hora(df, hora):
    horas=datos_hora(df, hora)

    parametros = expon.fit(horas)
    print('\nParámetros de modelo de ajuste: \n')
    print('loc = {:0.4f}, \nscale = {:0.4f}'.format(parametros[0], parametros[1]))

    f=Fitter(consumo_hora1)
    f.fit()
    f.get_best()
    f.summary()

    
#################################################
#PUNTO 4

'''
Función 4, de la misma forma recibe como parámetros el data frame y la hora de interés.
En este caso se llama a la función 2 para agrupar los datos y otener las estadísticas de estos.
Retorna la impresión de todas las estadísticas solicitadas.
'''

#En esta función de la misma forma que el punto anterior se llama a la función datos-hora para tener los datos del consumo
#Una vez con los datos del consumo guardado se proceden a sacar las estadísticas con las bibliotecas correspondientes
#Una vez calculdas, se guardan en las respectivas variables y se imprimen
def estadisticas_hora(df, hora):
    datos = datos_hora(df, hora)
    media=datos.mean()
    varianza=datos.var()
    desviacion=consumo_hora1.std()
    inclinacion=skew(datos)
    kurtosis=consumo_hora1.kurt()

    print("La media: ", media)
    print("La varianza: ", varianza)
    print("La desviacion estandar: ", desviacion)
    print("La inclinacion: ", inclinacion)
    print("La kurtosis: ", kurtosis)


#################################################
#PUNTO 5

'''
Función 5, de la misma forma se reciben como parámetros un data frame y la hora, 
continuando con la metodología de los puntos anteriores guardar los datos de la potencia
en un vector, y en este caso posteriormente graficarlo en un histograma.
'''

#De la misma forma se guardan los datos de consumo de una hora en la variable datos
#Una vez teniendo los datos, se realiza un histograma de estos
#Se imprime el histograma
def visualizacion_hora(df, hora):

    datos=datos_hora(df, hora)
    plt.hist(datos, color="blue")
    plt.title("Distribucion de potencia a las: " + str(hora))
    plt.xlabel('Potencia (MW)')
    plt.ylabel('Cantidad de días')
    plt.show()


#################################################
#PUNTO 6

'''
Función 6, se recibe de nuevo como parámetro el data frame, y en este caso
también se reciben dos diferentes horas, ya que el punto consiste en determinar la correlación
que existe en el consumo de potencia de estas dos horas.
Se imprime como resultado final el coeficiente de correlación.
'''

#Se llama a la funcion de consumo de horas dos veces, para las dos horas correspondientes
#Con la bibliotca pandas se calcula el coeficiente de correlación de estos dos conjuntos de datos obtenidos
def correlacion_horas(df, hora1, hora2):
    consumo1=datos_hora(datox, hora1)
    consumo2=datos_hora(datox, hora2)
    coef = np.corrcoef (consumo1, consumo2) [0,1]
    print("El coeficiente de correlacion es: ", coef)

    

#################################################
#PUNTO 7

'''
Función 7, prácticamente como continuando con el punto anterior, se reciben dos
horas distintas para poder llamar a la función de punto anterior, y en este caso
se determina la correlación de estas dos horas pero de forma gráfica.
Se imprime el histograma bivariado o en 3D de estos grupos de horas.
'''

#Se crea un histograma bivariado para determinar gráficamente la correlación de los dos conjuntos de datos
#Se definen los 3 ejes para el histograma 3D
#De la misma forma se definen los soportes de las barras de este gráfico
#Se definen  los arreglos con las dimensiones correspondientes para las barras
#Finalmente se imprime el gráfico 3D que muestra la correlación de horas
def visualizacion_horas(hora1, hora2):
    
    #Se definen los ejes para el histograma 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = hora1
    y = hora2
    hist, xedges, yedges = np.histogram2d(x, y, bins=20)

    # Se definen los soportes para las barras que va a tener el grafico
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.5, yedges[:-1] + 0.5, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Se definen arreglos con las dimensiones de las barras
    #Se imprime el histograma 3D
    dx = dy = 15 * np.ones_like(zpos)
    dz = hist.ravel()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    plt.xlabel('Potencia hora 1 (MW)')
    plt.ylabel('Potencia hora 2 (MW)')
    plt.show()


#################################################
#PUNTO 8

'''
Función 8, se reciben como parámetros el data frame, y dos valores, de punto inicial y final, esto
para delimitar el data frame al consumo en horas de una semana.
Se retorna la energía final calculada con la integral de trapz.
'''

#Se crea la funcion de energía semanal
#Se reciben como parámetros el data frame, y los puntos inicial y final para hacer el cálculo de potencias
#Una vez guardadas las potencias en el vector potencia se crea un vector de 168 posiciones para las 168 horas que 
#hay que integrar
#Teniendo los dos vectores para la integración se realiza esta con trapz y se retorna el vector con las 168 energías
def energia_semanal(df, inicio, final):
    df2=df.iloc[inicio:final]
    potencia = df2['MW']
    horas = []
    for i in range(168):
        vector_horas=horas.append(i)
    
    energia = np.trapz(potencia, vector_horas)

    return energia



##################################################
#PUNTO 9

'''
Función 9, se utiliza la función del punto anterior para calcular la energía de una semana
pero en este caso para recorrer el data frame en grupos de 168 horas, es decir, para ir por
todo el data frame de semana en semana, y de esta forma obtener las 52 energías para las 52 semanas.
También se establece la función para calcular de forma automática el modelo de mejor ajuste,
con sus parámetros y graficarlos.
'''

#Primero se define una función que se encarga de calcular las energías para todas las semanas con el uso 
#de la función energia-semanal
#La función va recorriendo el df de 168 en 168 filas, lo que equivale a semana por semana
#Va guardando la energía de cada semana en un vector deslizante
#Finalmente retorna las 52 energías semanales en un vector de esta dimensión
def consumo_energias_semanales(df):
    energias_semanales = np.empty(52)
    medias_semanales = np.empty(52)
    for i in range(52):
        energias_semanales[i] = energia_semanal(df,i*168,(i+1)*168)

    return energias_semanales


#Una vez teniendo el vector con las 52 energías semanales de la función anterior
#se procede a determinar el mejor ajuste con Fitter, al igual que se hizo en el punto 3
#Finalmente imprime el mejor ajuste junto con otros de los más aproximados
def modelo_energia_semanal(df):
    energias = consumo_energias_semanales(df)
    parametros = expon.fit(energias)
    print('\nParámetros de modelo de ajuste: \n')
    print('loc = {:0.4f}, \nscale = {:0.4f}'.format(parametros[0], parametros[1]))

    f=Fitter(energias)
    f.fit()
    f.get_best()
    f.summary()


##################################################

'''
Función 10, finalmente en esta función se utiliza el conjunto de 52 valores de energía
obteniéndolos llamando a la función anterior dentro de esta, para posteriormente, obtener
un modelo de mejor ajuste de distribución, pero en este caso por medio del teorema
de límite central.
Se obtienen los parámetros de importancia, se aplica el teorema, y se grafica la función del modelado.
'''

#Primero se guardan los datos de las energías semanales obtenidas con la función creada anteriormente
#Se define el número de factores que requiere el teorema de ímite central N-52
#Se calcula la media de los datos de energías obtenidos
#Se calcula la varianza de estos datos
#Se multiplican estos parámetros por N, como lo indica el teorema
#Una vez teniendo estos parámetros, se obtiene su función normalizada con stats.norm
#Se define un eje x con linspace, y la gaussiana determinada
#Finalmente se grafica, la distribución normal de estos datos con respecto al eje x creado
def modelo_energia_anual(df):
    datos = consumo_energias_semanales(df)
    N = 52
    media = datos.mean()
    miu = media * N
    estandar = np.std(datos)
    sigma = (estandar*estandar)
    gauss = stats.norm(miu, sigma)
    x = np.linspace(gauss.ppf(0.01), gauss.ppf(0.99), 100)
    plt.plot(x,norm.pdf(x, miu, sigma))
    plt.xlabel('Energía (MWh)')
    plt.ylabel('Ocurrencias')
    plt.show()

print("******* Analisis de datos del consumo de energía de la población en el año 2019 *******")
print("La horas asignadas son:", hora1, "y", hora2)

print("El mejor modelo de ajuste para las 11:00 fue Laplace asymmetric y se muestra a continuacion: ")
#modelo_hora(datox, hora1)
print("El mejor modelo de ajuste para las 15:00 fue Laplace asymmetric y se muestra a continuacion: ")
#modelo_hora(datox, hora2)
print(" ")
print("***Las stats par las 11:00 son: ****")
#estadisticas_hora(datox, hora1)
print(" ")
print("El histograma de distribución de consumo de potencia para las 11:00 se muestra a continuación")
#visualizacion_hora(datox, hora1) #se puede probar para cualquier hora, intentar con hora2 u hora3
print(" ")
print("El coeficiente de correlación es:", correlacion_horas(datox, hora1, hora2))
print(" ")
print("El histograma 3D que representa la correlación de las horas es el siguiente:")
visualizacion_horas(consumo_hora1, consumo_hora2)
print(" ")
print("La energía consumida por ejemplo en la primera semana es: ",energia_semanal(datox, 0, 168))
print("El modelo de mejor ajuste para la energía consumida en 52 semanas se muestra a continuación")
#modelo_energia_semanal(datox)
print(" ")
print("El modelo probabilístico de mejor ajuste que representa la energía anual mediante el teorema de límite central es: ")
#modelo_energia_anual(datox)