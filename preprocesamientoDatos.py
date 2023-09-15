class PreprocesamientoTitanic():
    def __init__(self) -> None:
        """
        La función inicializa una variable llamada "__df" en Ninguno.
        """
        self.__df   = None   
    
    def cargarDataset(self, pathDs):
        """
        La función carga un conjunto de datos desde una ruta determinada y lo almacena en un objeto
        Pandas DataFrame.
        
        :param pathDs: El parámetro "pathDs" es una cadena que representa la ruta al archivo del
        conjunto de datos que desea cargar. Debe ser la ruta del archivo, incluido el nombre y la
        extensión del archivo
        """
        self.__df = pd.read_csv(pathDs, encoding='utf-8')

    def mostrarRegistros(self, mostrar=5):
        """
        La función `mostrarRegistros` imprime el primer número de filas `mostrar` de un dataframe si
        `mostrar` es positivo, o el último número de filas `mostrar` si `mostrar` es negativo.
        
        :param mostrar: El parámetro "mostrar" se utiliza para especificar el número de registros que se
        mostrarán. Si "mostrar" es un número positivo, mostrará los primeros registros "mostrar" del
        conjunto de datos. Si "mostrar" es un número negativo, mostrará el último "mostrar", defaults to
        5 (optional)
        """
        try:
            if mostrar >0:
                print(self.__df.head(mostrar))
            elif mostrar <0:
                mostrar = mostrar*-1
                print(self.__df.tail(mostrar))
        except ValueError as e:
            print("Valor no válido, pida los primeros registros del dataset desde 5 hasta 1 y lo últimos registros desde -5 hasta -1", e)
        
    def borrarDatosNulos(self, listaCols):
        """
        La función "borrarDatosNulos" elimina valores nulos de una lista de columnas determinada.
        
        :param listaCols: El parámetro "listaCols" es una lista de nombres de columnas
        """
        for i in listaCols:
            if i in self.__df.columns and self.__df[i].isna().any():
                self.__df.dropna(subset=i, how='any', inplace=True)
                print(f"Valores nulos en la columna '{i}' llenados con 0 correctamente.")
            else:
                print(f"No se ha encontrado la columna '{i}' o sus valores están vacíos, omitiendo...")

    def consultarTiposDatos(self):
        """
        La función "consultarTiposDatos" se utiliza para recuperar información sobre tipos de datos.
        """
        print("Tipos de datos:")
        print(self.__df.dtypes)
        print(self.__df.info)

    
    def borrarColumnasIrrelevantes(self, listaCols):
        """
        La función "borrarColumnasIrrelevantes" elimina columnas irrelevantes de una lista determinada.
        
        :param listaCols: El parámetro "listaCols" es una lista que contiene los nombres de las columnas
        que desea eliminar de un conjunto de datos o marco de datos
        """
        for i in listaCols:
            if i in self.__df.columns:
                self.__df.drop(columns=[i], inplace=True)
                print(f"Valores nulos en la columna '{i}' eliminados.")
            else:
                print(f"No se ha encontrado la columna {i}, omitiendo")

    def llenarDatosNulosMedidas(self, listaCols, medida:str="mediana"):
        """
        La función `llenarDatosNulosMedidas` llena valores nulos en una lista de columnas usando una
        medida específica (ya sea mediana u otra medida).
        
        :param listaCols: Una lista de columnas en un conjunto de datos que deben completarse con
            valores faltantes
        :param medida: El parámetro "medida" es una cadena que especifica el método que se utilizará
            para completar valores nulos en el conjunto de datos. Las opciones para este parámetro son
            "mediana" o cualquier otro método que quieras implementar.
        :Args:
            Opciones disponibles: 'mediana', 'moda'. 
        :type medida: str (optional)
        """


        for i in listaCols:
            if i in self.__df.columns  and self.__df[i].isna().any():
                if  pd.api.types.is_numeric_dtype(self.__df[i]):         
                    if medida == "mediana":
                        medicion        = self.__df[i].median()
                        self.__df[i]    = self.__df[i].fillna(medicion)
                    elif medida == "moda":
                        medicion     = self.__df[i].mode()[0]
                        self.__df[i] = self.__df[i].fillna(medicion)
                print(f"Valores nulos en la columna '{i}' llenados correctamente con {medicion}. ")
            else:   
                print(f"No se ha encontrado la columna '{i}' o sus valores están vacíos, omitiendo...")

    def llenarDatosNulosTextuales(self, col, Nombre="Roberto"):  #No funciona correctamente
        """
        La función "llenarDatosNulosTextuales" está destinada a llenar datos textuales nulos en una
        columna, pero no está funcionando correctamente.
        
        :param col: El parámetro "col" representa la columna en la que se deben completar los valores
        nulos textuales
        :param Nombre: El parámetro "Nombre" es una cadena que representa un nombre. El valor
        predeterminado está establecido en "Roberto", defaults to Roberto (optional)
        """
        for i in col:
            if i in self.__df.columns and self.__df[i].isna().any():
                if pd.api.types.is_string_dtype(self.__df[i]) or pd.api.types.is_object_dtype(self.__df[i]):
                    """
                    La función "guardarDatasetProcesado" guarda un conjunto de datos procesado en una ruta
                    especificada.
                    
                    :param pathGuardado: El parámetro "pathGuardado" es una cadena que representa la ruta donde se
                    guardará el conjunto de datos procesado
                    """
                    self.__df[i] = self.__df[i].fillna(Nombre)
                    print(f"Valores nulos en la columna '{i}' llenados con nombres aleatorios correctamente.")
            else:
                print(f"No se ha encontrado la columna '{i}' o sus valores están vacíos, omitiendo...")
        
    def guardarDatasetProcesado(self, pathGuardado):
        self.__df.to_csv(pathGuardado,index=False)

from typing import Union
import pandas as pd
from faker import Faker
import random
import pandas as pd
from diccionarios import *
rutas = Diccionarios()
rutaTitanic = rutas.diccionariosDS.get("rutaTitanic")
rutaTitanicProcesado = rutas.diccionariosDS.get("rutaTitanicProcesado")
df = PreprocesamientoTitanic()
lista= ["last", "first"	,"gender", "age", "class", "fare" , "embarked", "survived", "asasa"]
#listaNombres= ["Efrain"]
#colseparada= ["first"]
df.cargarDataset(rutaTitanic)
df.consultarTiposDatos()
df.mostrarRegistros(72)
#df.borrarDatosNulos(lista)
#df.llenarDatosNulos(lista)
#df.borrarColumnasIrrelevantes(lista)
df.llenarDatosNulosMedidas(lista, medida="moda")
#df.llenarDatosNulosTextuales(colseparada) #No funciona correctamente
#df.mostrarRegistros(72)

#df.mostrarRegistros(72)
df.guardarDatasetProcesado(rutaTitanicProcesado)
