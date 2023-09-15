def main():
    """
    Esta función ejecuta diferentes modelos de aprendizaje automático (árbol de decisión, K-vecinos más
    cercanos y regresión logística) en el conjunto de datos del Titanic.
    """
    from diccionarios import Diccionarios
    import pandas as pd
    from modeloDNTREE import ModeloDNTREE
    from modeloKNN import ModeloKNN
    from modeloRL import ModeloRL
    from os import system
    import time
    rutas = Diccionarios()
    df = pd.read_csv(rutas.diccionariosDS.get("rutaTitanicProcesado"))


    KNN = ModeloKNN(df)
    RL = ModeloRL(df)
    DNTREE = ModeloDNTREE(df)
    opciones = 0
    while opciones !=5:
        print("\n1. Ejecutar modelo de árboles de decisiones\n2. Ejecutar modelo de vecinos cercanos\n3. Ejecutar modelo de regresión logística\n4. Ejecutar todos los modelos\n5. Salir")
        opciones = int(input("\n¿Qué opción desea elegir? (coloque solo el número de la opción): "))
        if opciones <1 or opciones >5:
            system("cls")
            print("Por favor, elija una opción válida (entre 1 y 5).")
        elif opciones ==1:
            #DNTREE
            #MÉTODOS PARA EL MODELO DE DNTREE
            DNTREE.cargarDatos()
            DNTREE.correrModeloDNTREE()
            # Visualizar las gráficas
        elif opciones ==2:
            #KNN
            #MÉTODOS PARA EL MODELO DE KNN
            KNN.cargarDatos()
            KNN.correrModeloKNN()
        elif opciones ==3:
            #RL
            #MÉTODOS PARA EL MODELO DE REGRESIÓN LOGÍSTICA
            RL.cargarDatos()
            RL.correrModeloRL()
        elif opciones ==4:
            #DNTREE
            #MÉTODOS PARA EL MODELO DE DNTREE
            DNTREE.cargarDatos()
            DNTREE.correrModeloDNTREE()

            #KNN
            #MÉTODOS PARA EL MODELO DE KNN
            KNN.cargarDatos()
            KNN.correrModeloKNN()

            #RL
            #MÉTODOS PARA EL MODELO DE REGRESIÓN LOGÍSTICA
            RL.cargarDatos()
            RL.correrModeloRL()

        elif opciones ==5:
            print("Programa finalizado")
            time.sleep(1.7)
            system("cls")
            

if __name__ == "__main__":
    main()