from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
class ModeloRL():
    def __init__(self, df:Path):
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
        self.__df               = df
        self.__modelo_edad      = None
        self.__modelo_sexo      = None
        self.__RL               = "regresión logística"

    @property
    def getModeloEdad(self)->LogisticRegression:
        return self.__modelo_edad
    @property
    def getRLexo(self)->LogisticRegression:
        return self.__modelo_sexo

    def cargarDatos(self)->None:
        self.__dataEntradaX = self.__df[['gender', 'age', 'class']]
        self.__dataObjetivoY = self.__df['survived']

    def convertirDatosNumericos(self)->None:
        self.__dataEntradaX = pd.get_dummies(self.__dataEntradaX, columns=['gender', 'class'], drop_first=True)

    def dividirDatos(self)->None:
        self.__XTrain_sexo, self.__XTest_sexo, self.__yTrain_sexo, self.__yTest_sexo = train_test_split(
            self.__dataEntradaX, self.__dataObjetivoY, test_size=0.2, random_state=42)

        self.__XTrain_edad, self.__XTest_edad, self.__yTrain_edad, self.__yTest_edad = train_test_split(
            self.__dataEntradaX, self.__dataObjetivoY, test_size=0.2, random_state=123)

    # Parte 2: Modelo de regresión logística para el sexo
    def crearRLexoRL(self)->None:
        self.__modelo_sexo = LogisticRegression()

    def entrenarRLexoRL(self)->None:
        self.__modelo_sexo.fit(self.__XTrain_sexo, self.__yTrain_sexo)

    def evaluarRLexoRL(self)->None:
        y_pred_sexo = self.__modelo_sexo.predict(self.__XTest_sexo)
        precision_sexo = precision_score(self.__yTest_sexo, y_pred_sexo, pos_label="yes")
        exactitud_sexo = accuracy_score(self.__yTest_sexo, y_pred_sexo)
        print(f'Precisión del modelo de {self.__RL} para sexo: {precision_sexo}')
        print(f'Exactitud del modelo de {self.__RL} para sexo: {exactitud_sexo}')
        self._visualizarPrediccionesSexo()

    # Parte 3: Modelo de regresión logística para la edad
    def crearModeloEdadRL(self)->None:
        self.__modelo_edad = LogisticRegression()

    def entrenarModeloEdadRL(self)->None:
        self.__modelo_edad.fit(self.__XTrain_edad, self.__yTrain_edad)

    def evaluarModeloEdadRL(self)->None:
        y_pred_edad = self.__modelo_edad.predict(self.__XTest_edad)
        precision_edad = precision_score(self.__yTest_edad, y_pred_edad, pos_label="yes")
        exactitud_edad = accuracy_score(self.__yTest_edad, y_pred_edad)
        print(f'Precisión del modelo de {self.__RL} para edad: {precision_edad}')
        print(f'Exactitud del modelo de {self.__RL} para edad: {exactitud_edad}')
        self._visualizarPrediccionesEdad()

    def _visualizarPrediccionesEdad(self)->None:
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

        fig.update_layout(title=f'Comparación de Datos Reales vs Datos Predichos por Edad - Modelo {self.__RL.capitalize()}',
                          xaxis=dict(title='Edad'),
                          yaxis=dict(title='Cantidad'),
                          barmode='stack')

        fig.show()

    def _visualizarPrediccionesSexo(self)->None:
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

        fig.update_layout(title=f'Comparación de Datos Reales vs Datos Predichos por Género - Modelo {self.__RL.capitalize()}',
                        xaxis=dict(title='Género'),
                        yaxis=dict(title='Cantidad'),
                        barmode='stack')

        fig.show()
        
        # Parte 4: Visualización de sobrevivientes por clase social

    def visualizarSobrevivientesPorClase(self)->None:
        SobrevivientesXClase = self.__df.groupby(['class', 'survived']).size().reset_index(name='count')
        
        fig = px.scatter(SobrevivientesXClase, x='class', y='count', title='Sobrevivientes por Clase Social',
                         labels={'class': 'Clase Social', 'count': 'Cantidad'},
                         color='survived', symbol='survived')
        
        fig.show()

    def correrModeloRL(self):
        """
        La función "correrModeloRL" se utiliza para ejecutar un modelo RL (regresión logística).
        """
        self.convertirDatosNumericos()
        self.dividirDatos()
        self.crearRLexoRL()
        self.entrenarRLexoRL()
        self.evaluarRLexoRL()
        self.crearModeloEdadRL()
        self.entrenarModeloEdadRL()
        self.evaluarModeloEdadRL()
        # Visualizar sobrevivientes por clase social
        self.visualizarSobrevivientesPorClase()

