import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
from sklearn.neighbors import KNeighborsClassifier
class ModeloKNN():
    def __init__(self, pathDF):
        """
        La función inicializa varias variables y establece el valor de la variable __KNN en "KNN".
        
        :param pathDF: El parámetro `pathDF` es la ruta al DataFrame que se utilizará para la
        inicialización de la clase. Se utiliza para establecer el atributo `__df`, que representa el
        objeto DataFrame
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
        self.__modelo_edad      = None
        self.__modelo_sexo      = None
        self.__KNN               = "KNN"

    @property
    def getModeloEdad(self)->KNeighborsClassifier:
        """
        La función getModeloEdad devuelve una instancia de la clase KNeighborsClassifier.
        """
        return self.__modelo_edad
    
    @property
    def getModeloSexo(self)->KNeighborsClassifier:
        """
        La función getModeloSexo devuelve una instancia de la clase KNeighborsClassifier.
        """
        return self.__modelo_sexo

    def cargarDatos(self)->None:
        """
        La función "cargarDatos" se utiliza para cargar datos.
        """
        self.__dataEntradaX = self.__df[['gender', 'age', 'class']]
        self.__dataObjetivoY = self.__df['survived']

    def convertirDatosNumericos(self)->None:
        """
        La función "convertirDatosNumericos" convierte datos a formato numérico.
        """
        self.__dataEntradaX = pd.get_dummies(self.__dataEntradaX, columns=['gender', 'class'], drop_first=True)

    def dividirDatos(self)->None:
        """
        La función "dividirDatos" no proporciona un resumen de una frase.
        """
        self.__XTrain_sexo, self.__XTest_sexo, self.__yTrain_sexo, self.__yTest_sexo = train_test_split(
            self.__dataEntradaX, self.__dataObjetivoY, test_size=0.2, random_state=42)

        self.__XTrain_edad, self.__XTest_edad, self.__yTrain_edad, self.__yTest_edad = train_test_split(
            self.__dataEntradaX, self.__dataObjetivoY, test_size=0.2, random_state=123)

    # Parte 2: Modelo de KNN para el sexo
    def crearModeloSexoKNN(self,nroVecinos = 5):
        """
        La función "crearModeloSexoKNN" crea un modelo KNN para predecir el género con un número
        específico de vecinos.
        
        :param nroVecinos: El parámetro "nroVecinos" es un parámetro opcional que especifica el número
        de vecinos a considerar al realizar predicciones utilizando el algoritmo K-Vecinos más cercanos.
        De forma predeterminada, está establecido en 5, lo que significa que el algoritmo considerará
        los 5 vecinos más cercanos a un punto de datos determinado cuando, defaults to 5 (optional)
        """
        self.__modelo_sexo = KNeighborsClassifier(n_neighbors=nroVecinos)

    def entrenarModeloSexoKNN(self)->None:
        """
        La función entrenarModeloSexoKNN entrena un modelo de K-Vecinos más cercanos para predecir el
        género.
        """
        self.__modelo_sexo.fit(self.__XTrain_sexo, self.__yTrain_sexo)

    def evaluarModeloSexoKNN(self)->None:
        """
        La función `evaluarModeloSexoKNN` no está definida correctamente.
        """
        y_pred_sexo = self.__modelo_sexo.predict(self.__XTest_sexo)
        precision_sexo = precision_score(self.__yTest_sexo, y_pred_sexo, pos_label="yes")
        exactitud_sexo = accuracy_score(self.__yTest_sexo, y_pred_sexo)
        print(f'Precisión del modelo de {self.__KNN} para sexo: {precision_sexo}')
        print(f'Exactitud del modelo de {self.__KNN} para sexo: {exactitud_sexo}')
        self._visualizarPrediccionesSexo()

    # Parte 3: Modelo de KNN para la edad
    def crearModeloEdadKNN(self,nroVecinos = 5):
        """
        La función "crearModeloEdadKNN" crea un modelo KNN para predecir la edad, con el número de
        vecinos especificado como parámetro opcional.
        
        :param nroVecinos: El parámetro "nroVecinos" es un parámetro opcional que especifica el número
        de vecinos a considerar al realizar predicciones utilizando el algoritmo K-Vecinos más cercanos.
        De forma predeterminada, está establecido en 5, pero puede cambiarlo a cualquier valor entero
        positivo, defaults to 5 (optional)
        """
        self.__modelo_edad = KNeighborsClassifier(n_neighbors=nroVecinos)

    def entrenarModeloEdadKNN(self)->None:
        """
        La función entrenarModeloEdadKNN entrena un modelo de K-Vecinos más cercanos para predecir la
        edad.
        """
        self.__modelo_edad.fit(self.__XTrain_edad, self.__yTrain_edad)

    def evaluarModeloEdadKNN(self)->None:
        """
        Esta función evalúa un modelo KNN para predecir la edad.
        """
        y_pred_edad = self.__modelo_edad.predict(self.__XTest_edad)
        precision_edad = precision_score(self.__yTest_edad, y_pred_edad, pos_label="yes")
        exactitud_edad = accuracy_score(self.__yTest_edad, y_pred_edad)
        print(f'Precisión del modelo de {self.__KNN} para edad: {precision_edad}')
        print(f'Exactitud del modelo de {self.__KNN} para edad: {exactitud_edad}')
        self._visualizarPrediccionesEdad()

    def _visualizarPrediccionesEdad(self)->None:
        """
        Esta función se utiliza para visualizar predicciones de edad.
        """
        dfReal = pd.DataFrame({'Edad': self.__XTest_edad['age'], 'Real': self.__yTest_edad})
        dfPredicciones = pd.DataFrame({'Edad': self.__XTest_edad['age'], 'Predicciones': self.__modelo_edad.predict(self.__XTest_edad)})
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
        fig.update_layout(title=f'Comparación de Datos Reales vs Datos Predichos por Edad - Modelo {self.__KNN.capitalize()}',
                          xaxis=dict(title='Edad'),
                          yaxis=dict(title='Cantidad'),
                          barmode='stack')

        fig.show()

    def _visualizarPrediccionesSexo(self)->None:
        """
        Esta función se utiliza para visualizar predicciones de género.
        """
        dfReal = pd.DataFrame({'Género': self.__XTest_sexo['gender_M'].replace({True: 'Masculino', False: 'Femenino'}), 'Real': self.__yTest_sexo})
        dfPredicciones = pd.DataFrame({'Género': self.__XTest_sexo['gender_M'].replace({True: 'Masculino', False: 'Femenino'}), 'Predicciones': self.__modelo_sexo.predict(self.__XTest_sexo)})
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
        fig.update_layout(title=f'Comparación de Datos Reales vs Datos Predichos por Género - Modelo {self.__KNN.capitalize()}',
                        xaxis=dict(title='Género'),
                        yaxis=dict(title='Cantidad'),
                        barmode='stack')
        fig.show()

    # Parte 4: Visualización de sobrevivientes por clase social
    def visualizarSobrevivientesPorClase(self)->None:
        """
        Esta función se utiliza para visualizar el número de supervivientes por clase.
        """
        SobrevivientesXClase = self.__df.groupby(['class', 'survived']).size().reset_index(name='count')
        fig = px.scatter(SobrevivientesXClase, x='class', y='count', title='Sobrevivientes por Clase Social',
                         labels={'class': 'Clase Social', 'count': 'Cantidad'},
                         color='survived', symbol='survived')
        
        fig.show()

    def correrModeloKNN(self, nroVecinos: int=5):
        """
        La función `correrModeloKNN` ejecuta un modelo de K vecinos más cercanos con un número
        específico de vecinos.
        
        :param nroVecinos: El parámetro "nroVecinos" es un número entero que representa el número de
        vecinos a considerar al realizar predicciones en el algoritmo K-Nearest Neighbors (KNN). Por
        defecto, está configurado en 5, lo que significa que el algoritmo considerará los 5 vecinos más
        cercanos a un determinado, defaults to 5
        :type nroVecinos: int (optional)
        """
        self.convertirDatosNumericos()
        self.dividirDatos()
        self.crearModeloSexoKNN(nroVecinos) #Esta ha sido la mejor cantidad de vecinos cercanos que encontré que generaban una mayor precisión y exactitud en el modelo
        self.entrenarModeloSexoKNN()
        self.evaluarModeloSexoKNN()
        self.crearModeloEdadKNN(nroVecinos) #Esta ha sido la mejor cantidad de vecinos cercanos que encontré que generaban una mayor precisión y exactitud en el modelo
        self.entrenarModeloEdadKNN()
        self.evaluarModeloEdadKNN()
        # Visualizar las gráficas
        self.visualizarSobrevivientesPorClase()

