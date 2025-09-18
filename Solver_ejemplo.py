"""
Tarea de la clase de Teoría de Grafos
Integrantes:
* Miguel Ángel González Echegollén
* Oliver de Jesús Martínez Pérez
* Carlos Alfonzo San Román
"""

import pandas as pd
import numpy as np
from random import shuffle
from IPython.display import display  # For Jupyter-like display in some environments

# Ejemplo de creación de tableros iniciales en Pandas
# Crearemos un tablero aleatorio en pandas
def crear_tablero_inicial(longitud=None, array=None): # si no se pasa array, se crea uno aleatorio    
    if array is not None: # Si se provee un arreglo, lo convertimos en DataFrame
        tablero = pd.DataFrame(
            array,
            index=[f"fila {i+1}" for i in range(array.shape[0])], # Asignar nombres a filas y columnas, shape 0 es filas, shape 1 es columnas
            columns=[f"col {j+1}" for j in range(array.shape[1])]
        )
        return tablero
    
    if longitud is None:
        raise ValueError("Debes proporcionar el tamaño del tablero")
    # Si se provee un arreglo, agregamos el hueco o espacio"Vacío" y lo convertimos en DataFrame
    numeros = list(range(1, longitud * longitud)) + ["Vacío"]
    shuffle(numeros)                                            # Mezclamos los números para crear un tablero aleatorio
    tablero = pd.DataFrame(
        np.array(numeros).reshape(longitud, longitud),
        index=[f"fila {i+1}" for i in range(longitud)],
        columns=[f"col {j+1}" for j in range(longitud)]
    )
    return tablero

# Ejemplo de uso
# Tablero 3x3 aleatorio
# tablero_3x3 = crear_tablero_inicial(longitud=3) # Aleatorio
tablero_3x3 = crear_tablero_inicial(array=np.array([[2, 8, 3],
                                                     [1, 6, 4],
                                                     [7, "Vacío", 5]]))
print("\nTablero 3x3:")
print(tablero_3x3)
#%%
variable = np.array([[2, 8, 3],
                 [1, 6, 4],
                 [7, "Vacío", 5]])
print(variable)
type(variable)

print(f"version numpy: {np.__version__}") # Output version numpy: 2.3.3
print(f"version pandas: {pd.__version__}") # Output version pandas: 2.3.2
#%%
# Funcion para crear el tablero al que queremos llegar, ojo se debe pasar un numpy array
def crear_tablero_objetivo(tablero_objetivo=None): 
    if tablero_objetivo is None:
        raise ValueError("Se debe proporcionar un tablero")
    
    # Convertir el tablero de numpy array a DataFrame de pandas
    tablero_objetivo = pd.DataFrame(tablero_objetivo)
    # Asignar nombres a filas y columnas
    tablero_objetivo.index = [f"fila {i+1}" for i in range(tablero_objetivo.shape[0])]
    tablero_objetivo.columns = [f"col {j+1}" for j in range(tablero_objetivo.shape[1])]
    return tablero_objetivo

# Ejemplo de uso
# DEfinimos un tablero objetivo
tablero_objetivo = np.array([[1, 2, 3],
                            [8, "Vacío", 4],
                            [7, 6, 5]])

tablero_objetivo = crear_tablero_objetivo(tablero_objetivo=tablero_objetivo)
print("\nTablero Objetivo:")
print(tablero_objetivo)
#%%
# Clase base del puzzle donde se definen las operaciones básicas
class PuzzleBase:
    def __init__(self, tablero_objetivo, tablero_inicial=None, profundidad=0):
        if tablero_inicial is None:                                 # Si no se proporciona un tablero inicial, se crea uno aleatorio
            self.filas, self.columnas = tablero_objetivo.shape
            self.tablero_inicial = crear_tablero_inicial(self.filas, self.columnas)
        else:
            self.tablero_inicial = tablero_inicial

        self.tablero_actual = self.tablero_inicial.copy()           # Tablero actual es el que vamos modificando
        self.tablero_objetivo = tablero_objetivo
        self.profundidad = profundidad                              # Profundidad en el árbol de búsqueda, es un parametro de g(u)
        self.evaluacion = self.evaluar_estado(print_eval=False)     # Evaluación del estado actual, es un parametro de h(u)
        self.spacePosition = self.encontrar_vacio()                 # Posición del espacio vacío
        self.mejorHijo = dict(
            {'Izquierdo': None,
             'Derecho': None,
             'Arriba': None,
             'Abajo': None})

    def encontrar_vacio(self, tablero=None):
        if tablero is None:
            posicion = np.where(self.tablero_actual.values == "Vacío")  # si no se pasa tablero, se usa el actual, que vamos modificando
        else:
            posicion = np.where(tablero.values == "Vacío")              # si se pasa tablero, se usa el que se pase (usado para el objetivo)

        return (posicion[0][0], posicion[1][0])                         # fila, columna

    def intercambiar(self, nueva_posicion):                         # nueva_posicion es una tupla (fila, columna)
        fila, columna = nueva_posicion
        fila_vacio, columna_vacio = self.spacePosition              # posición actual del espacio vacío
        if 0 <= fila < self.tablero_actual.shape[0] and 0 <= columna < self.tablero_actual.shape[1]: # Verificar límites
            val_nueva = self.tablero_actual.iloc[fila, columna]     # Valor en la nueva posición
            val_vacio = self.tablero_actual.iloc[fila_vacio, columna_vacio] # Valor en la posición del vacío (siempre "Vacío")
            # Intercambiar valores
            self.tablero_actual.iloc[fila_vacio, columna_vacio], self.tablero_actual.iloc[fila, columna] = val_nueva, val_vacio
            self.spacePosition = (fila, columna)                    # Actualizar posición del espacio vacío
            print(f"Intercambiando {val_nueva} con {val_vacio}")
            self.evaluar_estado(print_eval=True)                    # Re-evaluar el estado después del intercambio
        else:
            print("Posición fuera de los límites del tablero")

    def evaluar_estado(self, print_eval=False):
        # Create a mask that is True where the values in either dataframe are NOT "Vacío"
        pos_actual_vacio = self.encontrar_vacio()
        pos_original_vacio = self.encontrar_vacio(tablero=self.tablero_objetivo)
        if pos_actual_vacio == pos_original_vacio:                  # Si la posición del vacío es la misma en ambos tableros, no contarla para que no afecte la evaluación
            not_vacio_mask = 0
        else:
            not_vacio_mask = 1
        
        mask = (self.tablero_actual != self.tablero_objetivo)
        resultado = mask.sum().sum() - not_vacio_mask       # es igual a np.sum(mask.values) - not_vacio_mask
        if print_eval:
            print("El tablero está resuelto" if resultado == 0 else f"Función de evaluación f(u):= {self.profundidad} g(u) + {resultado} h(u) = {self.profundidad + resultado}")
        return mask

    def mostrar_tablero(self, tipo=None):   # tipo puede ser "inicial", "objetivo" o None (actual)
        if tipo == "inicial":
            print("\nTablero Inicial:")
            self.evaluacion = self.evaluar_estado(print_eval=True)
            print(self.tablero_inicial)
        elif tipo == "objetivo":
            print("\nTablero Objetivo:")
            print(self.tablero_objetivo)
        else:
            print("\nTablero Actual:")
            self.evaluacion = self.evaluar_estado(print_eval=True)
            print(self.tablero_actual)

    # Resaltar celdas que están correctas en verde y las incorrectas en rojo
    def resaltar_celda(self, df_actual, df_objetivo):
        def color_cells(row_idx, col_idx):                  # Función para determinar el color de fondo de cada celda
            val = df_actual.iloc[row_idx, col_idx]          # Comparamos el df actual
            objetivo = df_objetivo.iloc[row_idx, col_idx]   # con el df objetivo, que no cambia
            if val == "Vacío":
                return 'background-color: black'
            if val != objetivo:
                return 'background-color: darkred'
            return 'background-color: darkgreen'
        
        estilos = pd.DataFrame(
            [
                [color_cells(fila, col) for col in range(df_actual.shape[1])]   # Hacemos una lista de listas con los estilos, como insert sort, fila por fila y columna por columna
                for fila in range(df_actual.shape[0])
            ],
            index=df_actual.index,
            columns=df_actual.columns
        )
        resultados = df_actual.style.apply(lambda _: estilos, axis=None)        # Aplicar los estilos al DataFrame
        return resultados

# Clase que representa un nodo en el árbol de búsqueda del puzzle, hereda de PuzzleBase
class PuzzleTree(PuzzleBase):
    def __init__(self, tablero_objetivo, tablero_inicial=None, profundidad=0):
        # con super() llamamos al constructor de la clase base PuzzleBase
        super().__init__(tablero_objetivo, tablero_inicial, profundidad) # agregamos como argumentos el tablero objetivo, el inicial y la profundidad
        # Inicializamos los hijos como None, se crearán cuando se expanda el nodo
        self.hijo_izquierda = None
        self.hijo_derecha = None
        self.hijo_arriba = None
        self.hijo_abajo = None

    def tiene_hijo_izquierda(self):
        _, columna_vacio = self.spacePosition  # solo usamos la columna porque para izquierda y derecha solo importa la columna
        return columna_vacio > 0

    def tiene_hijo_derecha(self):
        _, columna_vacio = self.spacePosition
        return columna_vacio < self.tablero_actual.shape[1] - 1 # comprobamos que la columna del espacio vacío sea menor que el número de columnas - 1, o sea que no esté en la última columna

    def tiene_hijo_arriba(self):
        fila_vacio, _ = self.spacePosition                  # solo usamos la fila porque para arriba y abajo solo importa la fila
        return fila_vacio > 0                               # comprobamos que la fila del espacio vacío sea mayor que 0, o sea que no esté en la primera fila

    def tiene_hijo_abajo(self):
        fila_vacio, _ = self.spacePosition
        return fila_vacio < self.tablero_actual.shape[0] - 1 # comprobamos que la fila del espacio vacío sea menor que el número de filas - 1, o sea que no esté en la última fila

    # Expandir el nodo creando los hijos si es posible
    def add_depth(self):
        fila_vacio, columna_vacio = self.spacePosition
        next_depth = self.profundidad + 1

        if self.tiene_hijo_izquierda():
            tablero_izq = self.tablero_actual.copy()    # Hacemos una copia del tablero actual para el hijo izquierdo
            tablero_izq.iloc[fila_vacio, columna_vacio], tablero_izq.iloc[fila_vacio, columna_vacio - 1] = \
                tablero_izq.iloc[fila_vacio, columna_vacio - 1], tablero_izq.iloc[fila_vacio, columna_vacio]    # Intercambiamos el espacio vacío con el valor a la izquierda
            self.hijo_izquierda = PuzzleTree(self.tablero_objetivo, tablero_izq, next_depth)  # Creamos el hijo izquierdo con el nuevo tablero y la profundidad incrementada            
            print("Hijo izquierda:")
            self.hijo_izquierda.mostrar_tablero()
            contador = self.hijo_izquierda.evaluacion
            print(f"evaluacion: {contador.sum().sum()}")
        else: # si no sepuede crear el hijo, lo dejamos como estaba, None
            self.hijo_izquierda = None


        if self.tiene_hijo_derecha():
            tablero_der = self.tablero_actual.copy()
            tablero_der.iloc[fila_vacio, columna_vacio], tablero_der.iloc[fila_vacio, columna_vacio + 1] = \
                tablero_der.iloc[fila_vacio, columna_vacio + 1], tablero_der.iloc[fila_vacio, columna_vacio] # el +1 es para moverse a la derecha
            self.hijo_derecha = PuzzleTree(self.tablero_objetivo, tablero_der, next_depth)
            print("Hijo derecha:")
            self.hijo_derecha.mostrar_tablero()
            contador = self.hijo_derecha.evaluacion
            print(f"evaluacion: {contador.sum().sum()}")
        else:
            self.hijo_derecha = None


        if self.tiene_hijo_arriba():
            tablero_arr = self.tablero_actual.copy()
            tablero_arr.iloc[fila_vacio, columna_vacio], tablero_arr.iloc[fila_vacio - 1, columna_vacio] = \
                tablero_arr.iloc[fila_vacio - 1, columna_vacio], tablero_arr.iloc[fila_vacio, columna_vacio] # el -1 es para moverse hacia arriba
            self.hijo_arriba = PuzzleTree(self.tablero_objetivo, tablero_arr, next_depth)
            print("Hijo arriba:")
            self.hijo_arriba.mostrar_tablero()
            contador = self.hijo_arriba.evaluacion
            self.mejorHijo['Arriba'] = contador.sum().sum()
            print(self.mejorHijo)
            print(f"evaluacion: {contador.sum().sum()}")
        else:
            self.hijo_arriba = None


        if self.tiene_hijo_abajo():
            tablero_aba = self.tablero_actual.copy()
            tablero_aba.iloc[fila_vacio, columna_vacio], tablero_aba.iloc[fila_vacio + 1, columna_vacio] = \
                tablero_aba.iloc[fila_vacio + 1, columna_vacio], tablero_aba.iloc[fila_vacio, columna_vacio]
            self.hijo_abajo = PuzzleTree(self.tablero_objetivo, tablero_aba, next_depth)
            print("Hijo abajo:")
            self.hijo_abajo.mostrar_tablero()
            contador = self.hijo_abajo.evaluacion
            print(f"evaluacion: {contador.sum().sum()}")
        else:
            self.hijo_abajo = None


def main():
    # Entradas:
    tablero_objetivo = np.array([[1, 2, 3],
                                [8, "Vacío", 4],
                                [7, 6, 5]])

    tablero_objetivo = crear_tablero_objetivo(tablero_objetivo=tablero_objetivo)
    tablero_3x3 = crear_tablero_inicial(array=np.array([[2, 8, 3],
                                                        [1, 6, 4],
                                                        [7, "Vacío", 5]]))
    
    # tablero_3x3_imposible = crear_tablero_inicial(array=np.array([[1, 3, 7],
    #                                                             [8, "Vacío", 4],
    #                                                             [7, 6, 5]]))
    
    # Usar PuzzleTree para crear caso base
    puzzle = PuzzleTree(tablero_objetivo=tablero_objetivo,
                        tablero_inicial=tablero_3x3)

    # Mostrar los tableros
    print("\n" + "="*50)
    print("INICIO DEL PROGRAMA")
    print("="*50)
    puzzle.mostrar_tablero(tipo="inicial")
    puzzle.mostrar_tablero(tipo="objetivo")

    # Expandir hijos y mostrarlos
    print("\n" + "="*50)
    print("EXPANSIÓN DE NODOS")
    print("="*50)
    puzzle.add_depth()
    
    print("\n" + "="*50)
    print("EJECUCIÓN COMPLETADA")
    print("="*50)

if __name__ == "__main__":
    main()
