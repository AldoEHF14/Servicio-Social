import json
import os
import glob
from pathlib import Path

# Procesar un archivo JSON individual
def procesar_archivo_json(archivo):
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            datos = json.load(f)

            if 'History_Experiment_Simulated' not in datos:
                print(f"El archivo {archivo} no contiene la sección 'History_Experiment_Simulated'")
                return []

            historia = datos['History_Experiment_Simulated']
            initials = datos.get('profile', {}).get('initials', 'N/A')
            datos_procesados = []

            for ronda_idx, ronda in enumerate(historia):
                iteracion = ronda_idx + 1
                for diseno in ronda:
                    fila = [
                        initials,
                        iteracion,
                        diseno['time'],
                        diseno['risk'],
                        diseno['arrival'],
                        int(float(diseno['sliderDissatisfiedSatisfied'])),
                        int(float(diseno['sliderBoredExcited'])),
                    ]
                    datos_procesados.append(fila)
            return datos_procesados

    except FileNotFoundError:
        print(f"El archivo {archivo} no existe")
        return []
    except json.JSONDecodeError:
        print(f"Error al procesar el archivo {archivo}: formato JSON inválido")
        return []
    except KeyError as e:
        print(f'Error, falta el campo {e} en el archivo {archivo}')
        return []
    except Exception as e:
        print(f'Error inesperado al procesar el archivo {archivo}: {str(e)}')
        return []

# Escribir un archivo CSV
def escribir_csv(datos, archivoSalida, incluirEncabezado=True):
    try:
        with open(archivoSalida, 'w', encoding='utf-8') as f:
            if incluirEncabezado:
                encabezados = "initials iteracion time risk arrival valencia arousal"
                f.write(encabezados + '\n')

            for fila in datos:
                linea = ' '.join(str(valor) for valor in fila)
                f.write(linea + '\n')

        print(f"Archivo CSV creado exitosamente: {archivoSalida}")
    except Exception as e:
        print(f'Error al escribir el archivo CSV: {e}')

# Convertir archivo JSON individual
def convertir_archivo_individual(rutaArchivo, direcatorioSalida=None):
    datos = procesar_archivo_json(rutaArchivo)
    if not datos:
        print(f'No se encontraron datos para procesar')
        return
    nombreBase = Path(rutaArchivo).stem
    if direcatorioSalida:
        archivoSalida = os.path.join(direcatorioSalida, f'{nombreBase}.csv')
    else:
        archivoSalida = f'{nombreBase}.csv'
    escribir_csv(datos, archivoSalida)
    print(f'Procesados {len(datos)} diseños del archivo {rutaArchivo}')

# Convertir todos los archivos JSON en un directorio
def convertir_directorio_completo(directorio_entrada, archivo_salida_consolidado="resultados.csv"):
    patron = os.path.join(directorio_entrada, "*.json")
    archivos_json = glob.glob(patron)

    if not archivos_json:
        print(f"No se encontraron archivos JSON en {directorio_entrada}")
        return

    print(f"Encontrados {len(archivos_json)} archivos JSON para procesar")
    todos_los_datos = []
    archivos_procesados = 0

    for archivo in archivos_json:
        print(f"\nProcesando: {os.path.basename(archivo)}")
        datos = procesar_archivo_json(archivo)
        if datos:
            todos_los_datos.extend(datos)
            archivos_procesados += 1
            print(f"  - {len(datos)} diseños extraídos")

    if todos_los_datos:
        escribir_csv(todos_los_datos, archivo_salida_consolidado)
        print(f"\n{'='*50}")
        print(f"Resumen:")
        print(f"  - Archivos procesados: {archivos_procesados}/{len(archivos_json)}")
        print(f"  - Total de diseños: {len(todos_los_datos)}")
        print(f"  - Archivo de salida: {archivo_salida_consolidado}")
    else:
        print("No se encontraron datos para consolidar")

# Interfaz principal
def main():
    print("Conversor de archivos JSON a CSV")
    opcion = input("\n¿Qué deseas hacer?\n1. Convertir archivo individual\n2. Convertir directorio completo\nOpción: ")

    if opcion == '1':
        archivo = input("Ruta del archivo JSON: ")
        convertir_archivo_individual(archivo)
    elif opcion == "2":
        directorio = input("Ruta del directorio con archivos JSON: ")
        nombre_salida = input("Nombre del archivo para los resultados: ")
        if not nombre_salida:
            nombre_salida = "resultados.csv"
        else:
            nombre_salida = nombre_salida + ".csv"
        convertir_directorio_completo(directorio, nombre_salida)

if __name__ == "__main__":
    main()
