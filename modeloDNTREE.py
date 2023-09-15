from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
from sklearn import tree
class ModeloDNTREE():
    def __init__(self, pathDF:Path):
        """
        La función inicializa varias variables y establece el valor de una constante.
        
        :param pathDF: El parámetro `pathDF` es de tipo `Path` y representa la ruta al archivo de datos
        :type pathDF: Path
        """
        self.__dataEntradaX     = None
        self.__dataObjetivoY    = None
        self.__XTrain_sexo      = None
        self.__XTest_sexo       = None
        self.__yTrain_sexo      = None
        self.__yTest_sexo       = None
        self.__XTrain_edad      = None
        self.__XTest_edad       = None
        self.__yTrain_edad      = None
        self.__yTest_edad       = None
        self.__df               = pathDF
        self.__modeloEdad       = None
        self.__modeloSexo       = None
        self.__DNTREE           = "DNTREE"

    @property
    def getModeloEdad(self)->tree:
        """
        La función getModeloEdad devuelve el valor del atributo privado __modeloEdad.
        :return: El método está devolviendo el valor del atributo "__modeloEdad".
        """
        return self.__modeloEdad
    
    @property
    def getModeloSexo(self)->tree:
        """
        La función getModeloSexo devuelve un árbol.
        """
        return self.__modeloSexo

    def cargarDatos(self)->None:
        """
        La función "cargarDatos" carga los datos de las columnas 'género', 'edad' y 'clase' en la
        variable self.__dataEntradaX, y carga los datos de la columna 'sobrevivió' en la variable
        self.__dataObjetivoY.
        """
        self.__dataEntradaX = self.__df[['gender', 'age', 'class']]
        self.__dataObjetivoY = self.__df['survived']

    def convertirDatosNumericos(self)->None:
        """
        La función "convertirDatosNumericos" convierte variables categóricas en variables numéricas
        mediante codificación one-hot.
        """
        self.__dataEntradaX = pd.get_dummies(self.__dataEntradaX, columns=['gender', 'class'], drop_first=True)

    def dividirDatos(self)->None:
        """
        La función `dividirDatos` divide los datos de entrada y de destino en conjuntos de entrenamiento
        y prueba para las variables 'sexo' y 'edad'.
        """
        self.__XTrain_sexo, self.__XTest_sexo, self.__yTrain_sexo, self.__yTest_sexo = train_test_split(
            self.__dataEntradaX, self.__dataObjetivoY, test_size=0.2, random_state=42)

        self.__XTrain_edad, self.__XTest_edad, self.__yTrain_edad, self.__yTest_edad = train_test_split(
            self.__dataEntradaX, self.__dataObjetivoY, test_size=0.2, random_state=123)

    # Parte 2: Modelo de regresión DNTREE el sexo
    def crearModeloSexoDNTREE(self)->None:
        """
        La función "crearModeloSexoDNTREE" se utiliza para crear un modelo de árbol de decisión para
        predecir el género.
        """
        self.__modeloSexo = tree.DecisionTreeClassifier()

    def entrenarModeloSexoDNTREE(self)->None:
        """
        La función entrenarModeloSexoDNTREE entrena un modelo para predecir el género mediante árboles
        de decisión.
        """
        self.__modeloSexo.fit(self.__XTrain_sexo, self.__yTrain_sexo)

    def evaluarModeloSexoDNTREE(self)->None:
        """
        La función "evaluarModeloSexoDNTREE" no proporciona un resumen de una oración.
        """
        y_pred_sexo = self.__modeloSexo.predict(self.__XTest_sexo)
        precision_sexo = precision_score(self.__yTest_sexo, y_pred_sexo, pos_label="yes")
        exactitud_sexo = accuracy_score(self.__yTest_sexo, y_pred_sexo)
        print(f'Precisión del modelo de {self.__DNTREE} para sexo: {precision_sexo}')
        print(f'Exactitud del modelo de {self.__DNTREE} para sexo: {exactitud_sexo}')
        self._visualizarPrediccionesSexo()

    # Parte 3: Modelo de DNTREE para la edad
    def crearModeloEdadDNTREE(self)->None:
        """
        La función crearModeloEdadDNTREE crea un modelo de árbol de decisión para predecir la edad.
        
        :param self: El parámetro "self" es una referencia a la instancia actual de la clase. Se utiliza
        para acceder a los atributos y métodos de la clase. En este caso se utiliza para definir un
        método llamado "crearModeloEdadDNTREE" dentro de una clase
        :type self: None
        """
        self.__modeloEdad = tree.DecisionTreeClassifier()

    def entrenarModeloEdadDNTREE(self)->None:
        """
        La función "entrenarModeloEdadDNTREE" se utiliza para entrenar un modelo de predicción de edad
        mediante árboles de decisión.
        
        :param self: El parámetro "self" es una referencia a la instancia actual de la clase. Se utiliza
        para acceder a los atributos y métodos de la clase. En este caso se utiliza para definir un
        método llamado "entrenarModeloEdadDNTREE" dentro de una clase
        :type self: None
        """
        self.__modeloEdad.fit(self.__XTrain_edad, self.__yTrain_edad)

    def evaluarModeloEdadDNTREE(self)->None:
        """
        La función `evaluarModeloEdadDNTREE` no toma ningún argumento y no devuelve ningún valor.
        
        :param self: El parámetro "self" se refiere a la instancia de la clase a la que pertenece el
        método. Es una convención en Python utilizar "self" como primer nombre de parámetro en los
        métodos de instancia. Por convención, se denomina "yo", pero puedes elegir el nombre que desees.
        Es usado para
        :type self: None
        """
        y_pred_edad = self.__modeloEdad.predict(self.__XTest_edad)
        precision_edad = precision_score(self.__yTest_edad, y_pred_edad, pos_label="yes")
        exactitud_edad = accuracy_score(self.__yTest_edad, y_pred_edad)
        print(f'Precisión del modelo de {self.__DNTREE} para edad: {precision_edad}')
        print(f'Exactitud del modelo de {self.__DNTREE} para edad: {exactitud_edad}')
        self._visualizarPrediccionesEdad()

    def _visualizarPrediccionesEdad(self)->None:
        """
        Esta función se utiliza para visualizar predicciones de edad.
        
        :param self: El parámetro "self" se refiere a la instancia de la clase a la que pertenece el
        método. Se utiliza para acceder a los atributos y métodos de la clase. En este caso, el método
        "_visualizarPrediccionesEdad" es un método de una clase, y "self" se utiliza para
        :type self: None
        """
        dfReal = pd.DataFrame({'Edad': self.__XTest_edad['age'], 'Real': self.__yTest_edad})
        dfPredicciones = pd.DataFrame({'Edad': self.__XTest_edad['age'], 'Predicciones': self.__modeloEdad.predict(self.__XTest_edad)})
        recuentoReal = dfReal.groupby(['Edad', 'Real']).size().reset_index(name='Cantidad')
        recuentoPredicciones = dfPredicciones.groupby(['Edad', 'Predicciones']).size().reset_index(name='Cantidad')
        fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('Datos Reales', 'Datos Predichos'))
        barRealSobrevivio = go.Bar(x=recuentoReal[recuentoReal['Real'] == 'yes']['Edad'],
                                     y=recuentoReal[recuentoReal['Real'] == 'yes']['Cantidad'],
                                     name='Sobrevivió (Real)', marker_color='green')
        barRealNoSobrevivio = go.Bar(x=recuentoReal[recuentoReal['Real'] == 'no']['Edad'],
                                        y=recuentoReal[recuentoReal['Real'] == 'no']['Cantidad'],
                                        name='No Sobrevivió (Real)', marker_color='red')
        barPrediccionSobrevivio = go.Bar(x=recuentoPredicciones[recuentoPredicciones['Predicciones'] == 'yes']['Edad'],
                                         y=recuentoPredicciones[recuentoPredicciones['Predicciones'] == 'yes']['Cantidad'],
                                         name='Sobrevivió (Predicho)', marker_color='lightgreen')
        barPrediccionNoSobrevivio = go.Bar(x=recuentoPredicciones[recuentoPredicciones['Predicciones'] == 'no']['Edad'],
                                            y=recuentoPredicciones[recuentoPredicciones['Predicciones'] == 'no']['Cantidad'],
                                            name='No Sobrevivió (Predicho)', marker_color='lightcoral')
        fig.add_trace(barRealSobrevivio, row=1, col=1)
        fig.add_trace(barRealNoSobrevivio, row=1, col=1)
        fig.add_trace(barPrediccionSobrevivio, row=1, col=2)
        fig.add_trace(barPrediccionNoSobrevivio, row=1, col=2)
        fig.update_layout(title=f'Comparación de Datos Reales vs Datos Predichos por Edad - Modelo {self.__DNTREE.capitalize()}',
                          xaxis=dict(title='Edad'),
                          yaxis=dict(title='Cantidad'),
                          barmode='stack')
        fig.show()

    def _visualizarPrediccionesSexo(self)->None:
        """
        Esta función se utiliza para visualizar predicciones de género.
        
        :param self: El parámetro "self" se refiere a la instancia de la clase a la que pertenece el
        método. Permite que el método acceda y modifique los atributos y métodos de la clase. En este
        caso el método "_visualizarPrediccionesSexo" no toma ningún parámetro adicional
        :type self: None
        """
        dfReal = pd.DataFrame({'Género': self.__XTest_sexo['gender_M'].replace({True: 'Masculino', False: 'Femenino'}), 'Real': self.__yTest_sexo})
        dfPredicciones = pd.DataFrame({'Género': self.__XTest_sexo['gender_M'].replace({True: 'Masculino', False: 'Femenino'}), 'Predicciones': self.__modeloSexo.predict(self.__XTest_sexo)})
        recuentoReal = dfReal.groupby(['Género', 'Real']).size().reset_index(name='Cantidad')
        recuentoPredicciones = dfPredicciones.groupby(['Género', 'Predicciones']).size().reset_index(name='Cantidad')
        fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('Datos Reales', 'Datos Predichos'))
        barRealSobrevivio = go.Bar(x=recuentoReal['Género'],
                                    y=recuentoReal.apply(lambda row: row['Cantidad'] if row['Real'] == 'yes' else 0, axis=1),
                                    name='Sobrevivió (Real)', marker_color='green')
        barRealNoSobrevivio = go.Bar(x=recuentoReal['Género'],
                                        y=recuentoReal.apply(lambda row: row['Cantidad'] if row['Real'] == 'no' else 0, axis=1),
                                        name='No Sobrevivió (Real)', marker_color='red')
        barPrediccionSobrevivio = go.Bar(x=recuentoPredicciones['Género'],
                                        y=recuentoPredicciones.apply(lambda row: row['Cantidad'] if row['Predicciones'] == 'yes' else 0, axis=1),
                                        name='Sobrevivió (Predicho)', marker_color='lightgreen')
        barPrediccionNoSobrevivio = go.Bar(x=recuentoPredicciones['Género'],
                                            y=recuentoPredicciones.apply(lambda row: row['Cantidad'] if row['Predicciones'] == 'no' else 0, axis=1),
                                            name='No Sobrevivió (Predicho)', marker_color='lightcoral')
        fig.add_trace(barRealSobrevivio, row=1, col=1)
        fig.add_trace(barRealNoSobrevivio, row=1, col=1)
        fig.add_trace(barPrediccionSobrevivio, row=1, col=2)
        fig.add_trace(barPrediccionNoSobrevivio, row=1, col=2)
        fig.update_layout(title=f'Comparación de Datos Reales vs Datos Predichos por Género - Modelo {self.__DNTREE.capitalize()}',
                        xaxis=dict(title='Género'),
                        yaxis=dict(title='Cantidad'),
                        barmode='stack')
        fig.show()

    # Parte 4: Visualización de sobrevivientes por clase social
    def visualizarSobrevivientesPorClase(self)->None:
        """
        Esta función se utiliza para visualizar el número de supervivientes por clase.
        
        :param self: El parámetro "self" se refiere a la instancia de la clase en la que se llama al
        método. Es una convención en Python utilizar "self" como primer nombre de parámetro en los
        métodos de instancia
        :type self: None
        """
        SobrevivientesXClase = self.__df.groupby(['class', 'survived']).size().reset_index(name='count')
        fig = px.scatter(SobrevivientesXClase, x='class', y='count', title='Sobrevivientes por Clase Social',
                         labels={'class': 'Clase Social', 'count': 'Cantidad'},
                         color='survived', symbol='survived')
        
        fig.show()


    def correrModeloDNTREE(self):
        """
        La función "correrModeloDNTREE" se utiliza para ejecutar un modelo llamado DNTREE.
        """
        self.convertirDatosNumericos()
        self.dividirDatos()
        self.crearModeloSexoDNTREE()
        self.entrenarModeloSexoDNTREE()
        self.evaluarModeloSexoDNTREE()
        self.crearModeloEdadDNTREE()
        self.entrenarModeloEdadDNTREE()
        self.evaluarModeloEdadDNTREE()
        self.visualizarSobrevivientesPorClase()