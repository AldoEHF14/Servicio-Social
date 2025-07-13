import pandas as pd 
import numpy as np 
import csv 
import random
from collections import defaultdict

def eliminacionAleatoria(archivo_entrada):
     filas_t = defaultdict(list)
     with open(archivo_entrada, 'r', newline='', encoding='utf-8') as archivo:
        lector = csv.DictReader(archivo, delimiter=' ')
        encabezado = lector.fieldnames
        for fila in lector:
            filas_t[fila['time']].append(fila)
     filas_unicas = [random.choice(grupo) for grupo in filas_t.values()]
     return filas_unicas, encabezado
    

def filtrarDatos(archivo_entrada):
    filas_unicas = []
    ids_vistos = set()
    with open(archivo_entrada, mode='r', newline='', encoding='utf-8') as archivo:
        lector = csv.DictReader(archivo, delimiter=' ')
        encabezado = lector.fieldnames
        for fila in lector:
                if fila['time'] not in ids_vistos:
                    ids_vistos.add(fila['time'])
                    filas_unicas.append(fila)
    return filas_unicas, encabezado


def reescribirArchivo(archivo_salida, datos_nuevos, encabezado):
     with open(archivo_salida, mode='w', newline='', encoding='utf-8') as archivo:
          escribir = csv.DictWriter(archivo, fieldnames=encabezado, delimiter=' ')
          escribir.writeheader()
          escribir.writerows(datos_nuevos)
          print("Archivo reescrito con exito")
    



def main():
    print("Filtrar Datos Repetidos")
    archivo = input("\nIngresa la ruta del archivo CSV \n Ruta: ")
    print("\nIngresa el tipo de eliminacion que preferies.")
    opcion = input("\n1. Conservar el primer valor encontrado.\n2. Eliminacion aleatoria\nOpcion:")
    if opcion == "1" :
        datos_filtrados, encabezados = filtrarDatos(archivo)
        reescribirArchivo(archivo, datos_filtrados, encabezados)
    elif opcion == "2" : 
        datos_filtrados, encabezados = eliminacionAleatoria(archivo)
        reescribirArchivo(archivo, datos_filtrados, encabezados)
    else:
         print("No esta dentro de las opciones.")
    
        


if __name__ == "__main__":
    main()