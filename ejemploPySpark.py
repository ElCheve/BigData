from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

class ProcesadorDatosPySpark:
    def __init__(self, app_name):
        self.spark = SparkSession.builder.appName(app_name).getOrCreate()
    
    def crearDataFrame(self, schema, datos):
        return self.spark.createDataFrame(datos, schema=schema)
    
    def mostrarDataFrame(self, df):
        df.show()

schema = StructType([
    StructField("nombre", StringType(), True),
    StructField("edad", IntegerType(), True),
    StructField("ciudad", StringType(), True)
])

datos = [
    {"nombre": "Alice", "edad": 30, "ciudad": "Nueva York"},
    {"nombre": "Bob", "edad": 25, "ciudad": "Los Ángeles"},
    {"nombre": "Charlie", "edad": 35, "ciudad": "Chicago"}
]

procesador = ProcesadorDatosPySpark("EjemploPySpark")
df = procesador.crearDataFrame(schema, datos)
procesador.mostrarDataFrame(df)
