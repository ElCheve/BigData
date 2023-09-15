import os
class Diccionarios():
    def __init__(self) -> None:
        self.__pathArchivoActual = os.path.abspath(__file__)
        pass

    @property
    def diccionariosDS(self):
        dicc = {
            "rutaTitanic"               : os.path.join(self.__pathArchivoActual, '..', 'Fuentes de datos', 'Titanic.csv'),
            "rutaTitanicProcesado"      : os.path.join(self.__pathArchivoActual, '..', 'Fuentes de datos', 'TitanicProcesado.csv'),
            "rutaGuardadoModeloSexo"       : os.path.join(self.__pathArchivoActual, '..', 'modelos', 'modelo_sexo.pkl'),
            "rutaGuardadoModeloEdad"       : os.path.join(self.__pathArchivoActual, '..', 'modelos', 'modelo_edad.pkl'),
        }
        return dicc
    
