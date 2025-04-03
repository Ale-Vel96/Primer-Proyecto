import yfinance as yf
from datetime import datetime
import numpy as np
from scipy.stats import kurtosis, skew, norm, t
import pandas as pd

import cufflinks as cf
cf.set_config_file(offline=True)

import warnings
warnings.filterwarnings('ignore')


from scipy import stats
import matplotlib.pyplot as plt

# Obtener la fecha de hoy en formato YYYY-MM-DD
hoy = datetime.today().strftime('%Y-%m-%d')

df = yf.download('^DJI', start='2010-01-01', end=hoy, progress=False)

rendimiento_simple = df["Close"].pct_change().dropna()
#rendimiento_simple

media = np.mean(rendimiento_simple)
sesgo = skew(rendimiento_simple)
desviacion = np.std(rendimiento_simple)
exceso_curtosis = kurtosis(rendimiento_simple) - 3

estadisticas_simple = pd.DataFrame({'Media': [media],"Desviación":[desviacion] ,'Sesgo': [sesgo], 'Exceso de curtosis': [exceso_curtosis]},index=['Estadísticas'])

# Mostrar resultados
print(estadisticas_simple)

rendimiento_log = np.log(df["Close"]).diff().dropna()
#rendimiento_log

media_log = np.mean(rendimiento_log)
sesgo_log = skew(rendimiento_log)
desviacion_log = np.std(rendimiento_log)
exceso_curtosis_log = kurtosis(rendimiento_log) - 3

estadisticas_simple = pd.DataFrame({'Media': [media_log], 'Sesgo': [sesgo_log], "Desviación":[desviacion_log],'Exceso de curtosis': [exceso_curtosis_log]},index=['Estadísticas'])

# Mos<trar resultados
print(estadisticas_simple)


VaR_normal, ES_normal = [], []
VaR_t, ES_t = [], []
hVaR, hES = [], []
VaR_m, ES_m = [], []

for alfa in [0.95, 0.975, 0.99]:
    # Normal
    var = norm.ppf(1 - alfa, media, desviacion)
    es = rendimiento_simple[rendimiento_simple <= var].mean()
    if isinstance(es, (pd.Series, np.ndarray)):
        es = es.values[0]
    VaR_normal.append(var)
    ES_normal.append(es)

    # T-Student
    var = t.ppf(1 - alfa, len(rendimiento_simple) - 1) * desviacion + media
    if isinstance(var, (pd.Series, np.ndarray)):
        var = var.values[0]
    es = rendimiento_simple[rendimiento_simple <= var].mean()
    if isinstance(es, (pd.Series, np.ndarray)):
        es = es.values[0]
    VaR_t.append(var)
    ES_t.append(es)

    # Histórico
    var = rendimiento_simple.quantile(1 - alfa)
    if isinstance(var, (pd.Series, np.ndarray)):
        var = var.values[0]
    es = rendimiento_simple[rendimiento_simple <= var].mean()
    if isinstance(es, (pd.Series, np.ndarray)):
        es = es.values[0]
    hVaR.append(var)
    hES.append(es)

    # MonteCarlo
    n = 100000
    simulaciones = np.random.normal(media, desviacion, n)
    var = np.percentile(simulaciones, (1 - alfa) * 100)
    es = rendimiento_simple[rendimiento_simple <= var].mean()
    if isinstance(es, (pd.Series, np.ndarray)):
        es = es.values[0]
    VaR_m.append(var)
    ES_m.append(es)

# Crear DataFrame con los resultados
resultados = pd.DataFrame({
    "Nivel de Confianza": [0.95, 0.975, 0.99],
    "VaR_Normal": VaR_normal,
    "ES_Normal": ES_normal,
    "VaR_t": VaR_t,
    "ES_t": ES_t,
    "VaR_Historico": hVaR,
    "ES_Historico": hES,
    "VaR_MonteCarlo": VaR_m,
    "ES_MonteCarlo": ES_m
}).set_index("Nivel de Confianza")


from IPython.display import display # Asegúrate de importar display

# ... (tu código existente) ...

# Mostrar resultados como tabla
display(resultados)


def rolling_window(rendimiento, window=252):
    alfas = [0.95, 0.99]

    # Diccionario para almacenar resultados
    resultados = {
        "Fecha": [],
        "VaR_95_h": [], "TVaR_95_h": [],
        "VaR_99_h": [], "TVaR_99_h": [],
        "VaR_95_n": [], "TVaR_95_n": [],
        "VaR_99_n": [], "TVaR_99_n": [],
    }

    for t in range(window, len(rendimiento)):
      datos_window = rendimiento.iloc[t-window:t].dropna()
      if len(datos_window) < window:
        raise ValueError(f"No hay suficientes datos. Se requieren minimo {window}, pero solo hay {len(datos_window)}.")
      else:
        media = datos_window.mean()
        desviacion = datos_window.std()

        #Fecha actual y rendimiento real
        fecha_actual = rendimiento.index[t]
        rendimiento_real = rendimiento.iloc[t]

        #VaR y TVaR Histórico
        var_h95 = datos_window.quantile(1 - 0.95)
        var_h99 = datos_window.quantile(1 - 0.99)
        tvar_h95 = datos_window[datos_window <= var_h95].mean()
        tvar_h99 = datos_window[datos_window <= var_h99].mean()


        #VaR y TVaR Normal
        var_n95 = norm.ppf(1 - 0.95, media, desviacion)
        var_n99 = norm.ppf(1 - 0.99, media, desviacion)

        var_n95 = var_n95.item() if isinstance(var_n95, (np.ndarray, pd.Series)) else var_n95
        var_n99 = var_n99.item() if isinstance(var_n99, (np.ndarray, pd.Series)) else var_n99

        tvar_n95 = datos_window[datos_window <= var_n95].mean()
        tvar_n99 = datos_window[datos_window <= var_n99].mean()

        convertir = [var_h95, var_h99, tvar_h95, tvar_h99, tvar_n95, tvar_n99]
        convertir = [j.values[0] if isinstance(j, (pd.Series, np.ndarray)) and hasattr(j, 'values') else j for j in convertir]

        # Asignar nuevamente a las variables originales
        var_h95, var_h99, tvar_h95, tvar_h99, tvar_n95, tvar_n99 = convertir

        # Almacenar resultados
        resultados["Fecha"].append(fecha_actual)
        resultados["VaR_95_h"].append(var_h95)
        resultados["TVaR_95_h"].append(tvar_h95)
        resultados["VaR_99_h"].append(var_h99)
        resultados["TVaR_99_h"].append(tvar_h99)
        resultados["VaR_95_n"].append(var_n95)
        resultados["TVaR_95_n"].append(tvar_n95)
        resultados["VaR_99_n"].append(var_n99)
        resultados["TVaR_99_n"].append(tvar_n99)

    #Convertir los resultados en un DataFrame
    df_resultados = pd.DataFrame(resultados).set_index("Fecha")

    # Graficar resultados
    plt.figure(figsize=(12, 6))
    plt.plot(df_resultados.index, rendimiento.loc[df_resultados.index], label="Rendimiento Logarítmico", color="black")
    plt.plot(df_resultados.index, df_resultados["VaR_95_n"], label="VaR 95% (Normal)", linestyle="--", color="blue")
    plt.plot(df_resultados.index, df_resultados["VaR_99_n"], label="VaR 99% (Normal)", linestyle="--", color="red")
    plt.plot(df_resultados.index, df_resultados["TVaR_95_n"], label="TVaR 95% (Normal)", linestyle=":", color="blue")
    plt.plot(df_resultados.index, df_resultados["TVaR_99_n"], label="TVaR 99% (Normal)", linestyle=":", color="red")
    plt.plot(df_resultados.index, df_resultados["VaR_95_h"], label="VaR 95% (Normal)", linestyle="--", color="magenta")
    plt.plot(df_resultados.index, df_resultados["VaR_99_h"], label="VaR 99% (Normal)", linestyle="--", color="purple")
    plt.plot(df_resultados.index, df_resultados["TVaR_95_h"], label="TVaR 95% (Normal)", linestyle=":", color="magenta")
    plt.plot(df_resultados.index, df_resultados["TVaR_99_h"], label="TVaR 99% (Normal)", linestyle=":", color="purple")
    plt.legend()
    plt.title("Rollin window")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True)
    plt.show()

    return df_resultados


def violaciones(df_resultados, rendimiento):
  n = len(df_resultados)

  #contadores
  conteo_violaciones = {
      "Violaciones_VaR_95_h": 0, "Violaciones_VaR_99_h": 0,
      "Violaciones_TVaR_95_h": 0, "Violaciones_TVaR_99_h": 0,
      "Violaciones_VaR_95_n": 0, "Violaciones_VaR_99_n": 0,
      "Violaciones_TVaR_95_n": 0, "Violaciones_TVaR_99_n": 0
  }

  for fecha in df_resultados.index:
    rendimiento_real = rendimiento.loc[fecha]

    var_h95, var_h99 = df_resultados.loc[fecha, ["VaR_95_h", "VaR_99_h"]]
    tvar_h95, tvar_h99 = df_resultados.loc[fecha, ["TVaR_95_h", "TVaR_99_h"]]
    var_n95, var_n99 = df_resultados.loc[fecha, ["VaR_95_n", "VaR_99_n"]]
    tvar_n95, tvar_n99 = df_resultados.loc[fecha, ["TVaR_95_n", "TVaR_99_n"]]

    #Sumar violaciones
    conteo_violaciones["Violaciones_VaR_95_h"] += 1 if rendimiento_real.item() < var_h95 else 0
    conteo_violaciones["Violaciones_VaR_99_h"] += 1 if rendimiento_real.item() < var_h99 else 0
    conteo_violaciones["Violaciones_TVaR_95_h"] += 1 if rendimiento_real.item() < tvar_h95 else 0
    conteo_violaciones["Violaciones_TVaR_99_h"] += 1 if rendimiento_real.item() < tvar_h99 else 0
    conteo_violaciones["Violaciones_VaR_95_n"] += 1 if rendimiento_real.item() < var_n95 else 0
    conteo_violaciones["Violaciones_VaR_99_n"] += 1 if rendimiento_real.item() < var_n99 else 0
    conteo_violaciones["Violaciones_TVaR_95_n"] += 1 if rendimiento_real.item() < tvar_n95 else 0
    conteo_violaciones["Violaciones_TVaR_99_n"] += 1 if rendimiento_real.item() < tvar_n99 else 0

  #porcentaje
  porcentaje_violaciones = {k: (v / n) * 100 for k, v in conteo_violaciones.items()} #k violaciones por caso, v la suma

  # Crear DataFrame
  df_porcentaje_violaciones = pd.DataFrame([porcentaje_violaciones])

  return df_porcentaje_violaciones


r = rolling_window(rendimiento_log)

violaciones(r, rendimiento_log)

def var_volatilidad_movil(rendimiento, window=252):
    alfas = [0.95, 0.99]
    resultados = {
        "Fecha": [],
        "VaR_95_m": [],
        "VaR_99_m": []
    }

    for t in range(window, len(rendimiento)):
        datos_window = rendimiento.iloc[t-window:t].dropna()
        if len(datos_window) < window:
            raise ValueError(f"No hay suficientes datos. Se requieren mínimo {window}, pero solo hay {len(datos_window)}.")

        desviacion = datos_window.std()
        fecha_actual = rendimiento.index[t]

        # Cálculo de VaR con volatilidad móvil y distribución normal
        var_m95 = norm.ppf(1 - 0.95) * desviacion
        var_m99 = norm.ppf(1 - 0.99) * desviacion

        resultados["Fecha"].append(fecha_actual)
        resultados["VaR_95_m"].append(var_m95)
        resultados["VaR_99_m"].append(var_m99)

    df_resultados = pd.DataFrame(resultados).set_index("Fecha")


    plt.figure(figsize=(12, 6))
    plt.plot(df_resultados.index, rendimiento.loc[df_resultados.index], label="Rendimientos", color="black", alpha=0.6)
    plt.plot(df_resultados.index, df_resultados["VaR_95_m"], label="VaR 95%", linestyle="--", color="cyan")
    plt.plot(df_resultados.index, df_resultados["VaR_99_m"], label="VaR 99%", linestyle="--", color="red")
    plt.legend()
    plt.title("Volatilidad Móvil - Rolling Window 252 días")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.show()

    return df_resultados




def violaciones_var_movil(df_resultados, rendimiento):
    n = len(df_resultados)
    conteo_violaciones = {
        "Violaciones_VaR_95_m": 0,
        "Violaciones_VaR_99_m": 0
    }

    for fecha in df_resultados.index:
        rendimiento_real = rendimiento.loc[fecha].squeeze()
        rendimiento_real = rendimiento_real.item() if isinstance(rendimiento_real, (np.ndarray, pd.Series)) else rendimiento_real

        var_m95, var_m99 = df_resultados.loc[fecha, ["VaR_95_m", "VaR_99_m"]].values
        var_m95 = var_m95.item() if isinstance(var_m95, (np.ndarray, pd.Series)) else var_m95
        var_m99 = var_m99.item() if isinstance(var_m99, (np.ndarray, pd.Series)) else var_m99

        conteo_violaciones["Violaciones_VaR_95_m"] += 1 if rendimiento_real < var_m95 else 0
        conteo_violaciones["Violaciones_VaR_99_m"] += 1 if rendimiento_real < var_m99 else 0

    porcentaje_violaciones = {k: (v / n) * 100 for k, v in conteo_violaciones.items()}

    return pd.DataFrame([porcentaje_violaciones])


vol = var_volatilidad_movil(rendimiento_log)
vio = violaciones_var_movil(vol, rendimiento_log)

var_volatilidad_movil(rendimiento_log)