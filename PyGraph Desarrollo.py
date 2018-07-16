#### df.iloc[2:3, 1:2] = 4  = Son 1 minutos
##                            1440 datos
#### df.iloc[2:3, 1:2] = 20 = Son 5 minutos
##                            288  datos

import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
#plt.style.available
plt.style.use('ggplot')
import numpy as np
#from glob import glob as gg
import seaborn as sns
import datetime
from time import time
#import time
import scipy
import scipy.fftpack
from scipy import signal as sp
import matplotlib.gridspec as gridspec
import os
import fnmatch
import folium
import branca
from pyql.geo.continents import Continent
from pyql.geo.placefinder import PlaceFinder

directorio = '/home/javier/Documentos/Persona Importante/Javier Villanueva-Valle/Personal Repositorio/PyGraph/AWD Files'
#buscarAWD = gg('{}/*.AWD'.format(directorio))
#buscarCSV = gg('{}/*.csv'.format(directorio))
#for AWD in buscarAWD:
#    print(AWD)

#help(os.mkdir)
#os.mkdir('{}/{}'.format(directorio, nombre))


#########################################################################
archivos_AWD = []
for AWD in os.listdir(directorio):
    if fnmatch.fnmatch(AWD, '*.AWD'):
        archivos_AWD.append(AWD)
print(archivos_AWD)
############### PARA ARCHIVOS AWD #################################
tiempo_inicial = time()
df = 'Lucero.AWD'
df = '{}/{}'.format(directorio, df)
df = pd.read_csv(df, sep='\t')
nombre = df.columns[0].strip()
fecha = df.iloc[0, 0]
hora = df.iloc[1, 0]
intervalo = fecha + ' ' + hora
salida_sol = '06:00:00'
puesta_sol = '18:00:00'
frecu = str(int(df.iloc[2, 0]))
if frecu == '4':
    frecu = '1T'
    len_datos = 1440
elif frecu == '20':
    frecu = '5T'
    len_datos = 288
edad = int(df.iloc[3, 0])
serie_acti = df.iloc[4, 0]
sexo = df.iloc[5, 0]
if sexo == 'M':
    sexo = 'Masculino'
elif sexo == 'F':
    sexo = 'Femenino'
datos = df.iloc[6:, 0].str.strip(' M').values.astype(float)
#datos = df.iloc[6:, 0].str.lstrip()
#datos = df.iloc[6:, 0].str.rstrip()
#datos = datos.values.astype(float)
#datos = df.iloc[6:, 0].values.astype(float)
#datos = df.iloc[6:, 0].str.strip(' M').values.astype(float)
indice = pd.date_range(intervalo, periods=len(datos), freq=frecu)
datos = pd.DataFrame(datos, index=indice)
datos['Unformat'] = (datos.iloc[:, 0:1]*999)/datos.iloc[:, 0:1].max()# Convertido a 999
datos['Porcentaje'] = (datos.iloc[:, 0:1]*1)/datos.iloc[:, 0:1].max()# Convertido a Porcentaje
datos['Puntaje_Z'] = (datos.iloc[:, 0:1] - datos.iloc[:, 0:1].mean())/datos.iloc[:, 0:1].std(ddof=0)# Convertido Puntaje Z EJEMPLO(df.a - df.a.mean())/df.a.std(ddof=0)
datos = datos.rename(columns={datos.columns[0]:nombre})
sum_dias = round(datos.resample('1D').sum()/len_datos).min()*100
total_dias = len(datos.resample('1D'))
horas = pd.date_range(datetime.date.today(), periods=len_datos, freq=frecu)
ts = pd.Series(np.random.randn(len(horas)), index=horas)
horas_conv = ts.asfreq('6H', method='pad')
horas = pd.DataFrame(horas_conv.index).rename(columns={0:'Fecha_Hora'})
horas = pd.to_datetime(horas['Fecha_Hora']).dt.time
horas = list(horas)
if total_dias in np.arange(2, 11, 1):
    tamaño = (6, 10)
    titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
    local_flecha = 10
    local_texto = -100
elif total_dias in np.arange(11, 32, 1):
    tamaño = (10, 20)
    titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
    local_flecha = 5
    local_texto = -30
elif total_dias in np.arange(31, 91, 1):
    tamaño = (10, 40)
    titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
    local_flecha = 5
    local_texto = -30
elif total_dias in np.arange(91, 182, 1):
    tamaño = (10, 50)
    titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
    local_flecha = 10
    local_texto = -100
elif total_dias in np.arange(182, 366, 1):
    tamaño = (12, 180)
    titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
    local_flecha = 300
    local_texto = 100
datos['Fecha'] = datos.index
datos['Fecha'] = pd.to_datetime(datos['Fecha']).dt.date.to_frame()
datos['Hora'] = datos.index
datos['Hora'] = pd.to_datetime(datos['Hora']).dt.time.to_frame()
Un_día_más = datos['Fecha'].iloc[-1] + datetime.timedelta(days=1)
tabla_ori = datos.pivot_table(values=datos.columns[0], index='Hora', columns='Fecha')
tabla_ori_descrip = round(tabla_ori.describe(), 2)
tabla_unfo = datos.pivot_table(values=datos.columns[1], index='Hora', columns='Fecha')
tabla_unfo_descrip = tabla_unfo.describe()
tabla_porc = datos.pivot_table(values=datos.columns[2], index='Hora', columns='Fecha')
tabla_porc_descrip = tabla_porc.describe()
tabla_z = datos.pivot_table(values=datos.columns[3], index='Hora', columns='Fecha')
tabla_z_descrip = tabla_z.describe()
##### Generar carpeta propia ################
os.makedirs('{}/{}'.format(directorio, nombre), exist_ok=True)
print('Se ha creado la carpeta del actimetro {} bajo el nombre de "{}"'.format(serie_acti, nombre))
#################################################
##### Excel de Datos #####
writer = pd.ExcelWriter('{}/{}/{} {} {}.xlsx'.format(directorio, nombre, serie_acti, nombre, datetime.datetime.now()))
datos.to_excel(writer, '{}'.format(serie_acti))
tabla_ori.to_excel(writer, 'Original')
tabla_ori_descrip.to_excel(writer, 'Descriptivos_Original')
tabla_unfo.to_excel(writer, 'Unformat')
tabla_unfo_descrip.to_excel(writer, 'Descriptivos_Unformat')
tabla_porc.to_excel(writer, 'Porcentaje')
tabla_porc_descrip.to_excel(writer, 'Descriptivos_Porcentaje')
tabla_z.to_excel(writer, 'Puntaje_Z')
tabla_z_descrip.to_excel(writer, 'Descriptivos_Puntaje Z')
writer.save()
################################################################
#tabla1 = tabla.iloc[:, :]
#tabla2 = tabla.iloc[:, 1::2]
#tabla3 = tabla.iloc[:, 2::2]
#tablas = pd.concat([tabla1, tabla2, tabla3], axis=1)
################################################################
##### Actograma Datos Originales
tablas_ori = pd.concat([tabla_ori, tabla_ori.iloc[:, 1::2], tabla_ori.iloc[:, 2::2]], axis=1)
tablas_ori[Un_día_más] = np.nan
tablas_ori = tablas_ori.sort_index(axis=1)
tablas_ori.plot.area(subplots=True, layout=(total_dias, 2), color='k', figsize=tamaño, sharex=True, xlim=(tabla_ori.index.min(), tabla_ori.index.max()), ylim=(datos[datos.columns[0]].min(), datos[datos.columns[0]].max()+10), xticks=horas, yticks=[], rot='vertical', fontsize=15);plt.subplots_adjust(hspace=0);plt.subplots_adjust(wspace=0);plt.xlabel(' ');#plt.xlabel(' ')
plt.savefig('{}/{}/{}_Original_{}.png'.format(directorio, nombre, nombre, datetime.datetime.now()), bbox_inches='tight')
plt.show()
##### Actograma Datos Unformat
tablas_unfo = pd.concat([tabla_unfo, tabla_unfo.iloc[:, 1::2], tabla_unfo.iloc[:, 2::2]], axis=1)
tablas_unfo[Un_día_más] = np.nan
tablas_unfo = tablas_unfo.sort_index(axis=1)
tablas_unfo.plot.area(subplots=True, layout=(total_dias, 2), color='k', figsize=tamaño, sharex=True, xlim=(tabla_unfo.index.min(), tabla_unfo.index.max()), ylim=(datos[datos.columns[1]].min(), datos[datos.columns[1]].max()+5), xticks=horas, yticks=[], rot='vertical', fontsize=15);plt.subplots_adjust(hspace=0);plt.subplots_adjust(wspace=0);plt.xlabel(' ');#plt.xlabel(' ')
plt.savefig('{}/{}/{}_Unformat_{}.png'.format(directorio, nombre, nombre, datetime.datetime.now()), bbox_inches='tight')
plt.show()
##### Actograma Datos Porcentaje
tablas_porc = pd.concat([tabla_porc, tabla_porc.iloc[:, 1::2], tabla_porc.iloc[:, 2::2]], axis=1)
tablas_porc[Un_día_más] = np.nan
tablas_porc = tablas_porc.sort_index(axis=1)
tablas_porc.plot.area(subplots=True, layout=(total_dias, 2), color='k', figsize=tamaño, sharex=True, xlim=(tabla_porc.index.min(), tabla_porc.index.max()), ylim=(datos[datos.columns[2]].min(), datos[datos.columns[2]].max()+0.1), xticks=horas, yticks=[], rot='vertical', fontsize=15);plt.subplots_adjust(hspace=0);plt.subplots_adjust(wspace=0);plt.xlabel(' ');#plt.xlabel(' ')
plt.savefig('{}/{}/{}_Porcentaje_{}.png'.format(directorio, nombre, nombre, datetime.datetime.now()), bbox_inches='tight')
plt.show()
##### Actograma Datos Z
tablas_z = pd.concat([tabla_z, tabla_z.iloc[:, 1::2], tabla_z.iloc[:, 2::2]], axis=1)
tablas_z[Un_día_más] = np.nan
tablas_z = tablas_z.sort_index(axis=1)
tablas_z.plot(subplots=True, layout=(total_dias, 2), color='k', figsize=tamaño, sharex=True, xlim=(tabla_z.index.min(), tabla_z.index.max()), ylim=(datos[datos.columns[3]].min(), datos[datos.columns[3]].max()+1), xticks=horas, yticks=[], rot='vertical', fontsize=15);plt.subplots_adjust(hspace=0);plt.subplots_adjust(wspace=0);plt.xlabel(' ');#plt.xlabel(' ')
plt.savefig('{}/{}/{}_Z_{}.png'.format(directorio, nombre, nombre, datetime.datetime.now()), bbox_inches='tight')
plt.show()
##########################################
##### Promedio por Días ######
### Tabla Original
tablas_ori.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tablas_ori.mean(axis=1))+400, 400), label='Promedio', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}/{}_Original_Promedio_{}.png'.format(directorio, nombre, nombre, datetime.datetime.now()))
plt.show()
### Tabla Unformat
tabla_unfo.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla_unfo.mean(axis=1))+50, 50), label='Promedio', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}/{}_Unformat_Promedio_{}.png'.format(directorio, nombre, nombre, datetime.datetime.now()))
plt.show()
### Tabla Porcentaje
tabla_porc.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla_porc.mean(axis=1))+0.05, 0.05), label='Promedio', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}/{}_Porcentaje_Promedio_{}.png'.format(directorio, nombre, nombre, datetime.datetime.now()))
plt.show()
### Tabla Z
tabla_z.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(min(tabla_z.mean(axis=1)), max(tabla_z.mean(axis=1))+0.2, 0.2), label='Promedio', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}/{}_Z_Promedio_{}.png'.format(directorio, nombre, nombre, datetime.datetime.now()))
plt.show()
##### Sumatoria por Días ######
### Tabla Original
tablas_ori.sum(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tablas_ori.sum(axis=1))+10000, 10000), label='Sumatoria', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}/{}_Original_Sumatoria_{}.png'.format(directorio, nombre, nombre, datetime.datetime.now()))
plt.show()
### Tabla Unformat
tabla_unfo.sum(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla_unfo.sum(axis=1))+200, 200), label='Sumatoria', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}/{}_Unformat_Sumatoria_{}.png'.format(directorio, nombre, nombre, datetime.datetime.now()))
plt.show()
### Tabla Porcentaje
tabla_porc.sum(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla_porc.sum(axis=1))+0.2, 0.2), label='Sumatoria', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}/{}_Porcentaje_Sumatoria_{}.png'.format(directorio, nombre, nombre, datetime.datetime.now()))
plt.show()
### Tabla Z
tabla_z.sum(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(min(tabla_z.sum(axis=1)), max(tabla_z.sum(axis=1))+2, 2), label='Sumatoria', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}/{}_Z_promedio_{}.png'.format(directorio, nombre, nombre, datetime.datetime.now()))
plt.show()
tiempo_final = time()
tiempo_ejecución = tiempo_final - tiempo_inicial
print('Listo\nTiempo de Procesamiento = {}\n'.format(datetime.timedelta(seconds=round(tiempo_ejecución, 2))))
#################################################################
############################################################
tablas_ori.plot.area(subplots=True, layout=(total_dias, 2), color='k', figsize=tamaño, sharex=True, xlim=(tabla_ori.index.min(), tabla_ori.index.max()), ylim=(datos[datos.columns[0]].min(), datos[datos.columns[0]].max()+10), xticks=horas, yticks=[], rot='vertical', fontsize=15);plt.subplots_adjust(hspace=0);plt.subplots_adjust(wspace=0);plt.xlabel(' ');#plt.xlabel(' ')

tabla1 = datos.reset_index()
tabla1 = tabla1['index']
tabla1 = tabla1.asfreq('6H', method='pad')
horas1 = tabla1.drop_duplicates().groupby(tabla1.dt.hour)
horas1.plot()

for n, g in horas1:
    #print(n)
    #print(g)
    [i.axvspan(g.iloc[0], g.iloc[1], facecolor='g', alpha=0.5) for i in ax] 
    if g.iloc[0].hour == 24:
        break


1440/4
360/4
tabla2 = pd.date_range(datetime.date.today(), periods=24, freq='1H')
ts2 = pd.Series(np.random.randn(len(tabla2)), index=tabla2)
df2 = ts2.rename_axis('Hora').resample('6H').sum()
df3 = df2.reset_index()
df4 = df3['Hora'].dt.time
hora2 = df4.drop_duplicates().groupby(df4.dt.hour)

#hora2 = df2['Hora'].dt.time
#hora3 = hora2.resample('6H')

##################
for n, g in horas:
    print(n)
    print(g)
    [i.axvspan(g.iloc[0], g.iloc[1], facecolor='g', alpha=0.5) for i in ax] 
    if g.iloc[0].year == 2016:
        break
###########################################################






#tablas.info()
nombre_act = tablas.columns.names[0]
nombre_act1 = tablas.columns.levels[0][0]
nombre_act2 = tablas.columns.levels[0][1]
act1 = tablas.xs('Act1', axis=1)
act2 = tablas.xs('Act2', axis=1)
tablas.xs('Act1', axis=1).plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
tablas.xs('Act2', axis=1).plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
#writer = pd.ExcelWriter('{}/{} {} {}.xlsx'.format(directorio, serie_acti, nombre, datetime.datetime.now()))
#datos.to_excel(writer, '{}'.format(serie_acti))
#tabla.to_excel(writer, 'Por Hora')
#tabla_descrip.to_excel(writer, 'Descriptivos por día')
#writer.save()

###############################################################

###############################################################
#### DEFINIR GRAFICAS PARA *.AWD #################
tabla.plot.area(subplots=True, layout=(total_dias, 2), color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.subplots_adjust(hspace=0); plt.subplots_adjust(wspace=0);

tablas.plot.area(subplots=True, layout=(total_dias, 2), color='k', figsize=tamaño, sharex=True, xlim=(tablas.index.min(), tablas.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.subplots_adjust(hspace=0); plt.subplots_adjust(wspace=0);

##### Actograma 1 #####
act1.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(act1.index.min(), act1.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.xlabel(titulo, fontsize=tamaño_letra); plt.subplots_adjust(hspace=0);
plt.savefig('{}/{}_{}_{}.png'.format(directorio, nombre, nombre_act1, datetime.datetime.now()))
plt.show()
##### Actograma 2 #####
act2.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(act2.index.min(), act2.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.xlabel(titulo, fontsize=tamaño_letra); plt.subplots_adjust(hspace=0);
plt.savefig('{}/{}_{}_{}.png'.format(directorio, nombre, nombre_act2, datetime.datetime.now()))
plt.show()



############### PARA ARCHIVOS TELENAX CSV #################################

##################################################################
archivos_CSV = []
for CSV in os.listdir(directorio):
    if fnmatch.fnmatch(CSV, '*.csv'):
        archivos_CSV.append(CSV)
print(archivos_CSV)
##################################################################
df_csv = 'JairoH602_2018_05_14_11_53_48.csv'
nombre = 'Jairo'
dispo = 'H602'
salida_sol = '06:00:00'
puesta_sol = '18:00:00'
df = '{}/{}'.format(directorio, df_csv)
df = pd.read_csv(df, sep=',')
df = df.replace('N/A', np.nan)
df = df.replace('Unsuccessful GPS attempt', np.nan)
df = df.replace('BLUETOOTH', np.nan)
fecha = df['Date']
hora = df['Time']
intervalo = fecha + ' ' + hora
intervalo = pd.to_datetime(intervalo)
df = df.set_index(intervalo)
español = {'Date':'Fecha', 'Time':'Tiempo', 'Altitude':'Altitud', 'Latitude':'Latitud', 'Longitude':'Longitud', 'Speed':'Velocidad', 'No. Satelites':'#_Satelites', 'Activity Level':'Actividad', 'Temperature':'Temperatura', 'GPS Battery Voltage':'GPS Voltage de la Batería', 'Notification':'Notificación'}
df = df.rename(columns=español)
df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
df['Tiempo'] = pd.to_datetime(df['Tiempo']).dt.time
#elec_años = list(df.resample('1Y').nunique().index)
df = df[df.index >= '2018']
df['Altitud'] = df['Altitud'].str.strip(' mt')
df['Altitud'] = df['Altitud'].str.lstrip()
df['Altitud'] = df['Altitud'].str.rstrip().astype(float)
df['Velocidad'] = df['Velocidad'].str.strip(' m/s')
df['Velocidad'] = df['Velocidad'].str.lstrip()
df['Velocidad'] = df['Velocidad'].str.rstrip().astype(float)
df['Temperatura'] = df['Temperatura'].str.strip(' oC')
df['Temperatura'] = df['Temperatura'].str.lstrip()
df['Temperatura'] = df['Temperatura'].str.rstrip().astype(float)
sum_dias = round(df.loc[:, 'Actividad'].resample('1D').sum()).min()*100
total_dias = len(df.resample('1D'))
horas = pd.date_range(datetime.date.today(), periods=288, freq='5T')
ts = pd.Series(np.random.randn(len(horas)), index=horas)
horas_conv = ts.asfreq('6H', method='pad')
horas = pd.DataFrame(horas_conv.index).rename(columns={0:'Fecha_Hora'})
horas = pd.to_datetime(horas['Fecha_Hora']).dt.time
horas = list(horas)
if total_dias in np.arange(2, 31, 1):
    tamaño = (8, 15)
    #titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
    tamaño_letra = 15
    etiquetay = [0, 500, 1000]
    etiquetayprom = np.arange(0, 1100, 100)
    etiquetaysum = np.arange(0, sum_dias, 100)
    local_flecha = 10
    local_texto = -100
elif total_dias in np.arange(31, 91, 1):
    tamaño = (10, 40)
    titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
    tamaño_letra = 20
    etiquetay = [0, 2500]
    etiquetayprom = np.arange(0, 1100, 100)
    etiquetaysum = np.arange(0, sum_dias, 100)
    local_flecha = 5
    local_texto = -30
elif total_dias in np.arange(91, 182, 1):
    tamaño = (10, 50)
    #titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
    tamaño_letra = 25
    etiquetay = [0, 2500]
    etiquetayprom = np.arange(0, 1100, 100)
    etiquetaysum = np.arange(0, sum_dias, 100)
    local_flecha = 10
    local_texto = -100
elif total_dias in np.arange(182, 366, 1):
    tamaño = (12, 180)
    #titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
    tamaño_letra = 25
    etiquetay = [0, 1000]
    etiquetayprom = np.arange(0, 1100, 100)
    etiquetaysum = np.arange(0, sum_dias, 100)
    local_flecha = 300
    local_texto = 100
df['Unformat'] = (df.loc[:, 'Actividad']*999)/df.loc[:, 'Actividad'].max()# Convertido a 999
df['Porcentaje'] = (df.loc[:, 'Actividad']*1)/df.loc[:, 'Actividad'].max()# Convertido a Porcentaje
df['zscore'] = (df.loc[:, 'Actividad'] - df.loc[:, 'Actividad'].mean())/df.loc[:, 'Actividad'].std(ddof=0)# Convertido Puntaje Z EJEMPLO(df.a - df.a.mean())/df.a.std(ddof=0)
tabla_ori = df.pivot_table(values='Actividad', index='Tiempo', columns='Fecha')
tabla_ori_descrip = tabla_ori.describe()
tabla_unfo = df.pivot_table(values='Unformat', index='Tiempo', columns='Fecha')
tabla_unfo_descrip = tabla_unfo.describe()
tabla_porc = df.pivot_table(values='Porcentaje', index='Tiempo', columns='Fecha')
tabla_porc_descrip = tabla_porc.describe()
tabla_z = df.pivot_table(values='zscore', index='Tiempo', columns='Fecha')
tabla_z_descrip = tabla_z.describe()
tabla_temp = df.pivot_table(values='Temperatura', index='Tiempo', columns='Fecha')
tabla_temp_descrip = tabla_temp.describe()
##### Excel de Datos #####
#writer = pd.ExcelWriter('{}/{} {} {}.xlsx'.format(directorio, dispo, nombre, datetime.datetime.now()))
#df.to_excel(writer, '{}'.format(dispo))
#tabla_ori.to_excel(writer, 'Original')
#tabla_ori_descrip.to_excel(writer, 'Descriptivos_Original')
#tabla_unfo.to_excel(writer, 'Unformat')
#tabla_unfo_descrip.to_excel(writer, 'Descriptivos_Unformat')
#tabla_porc.to_excel(writer, 'Porcentaje')
#tabla_porc_descrip.to_excel(writer, 'Descriptivos_Porcentaje')
#tabla_z.to_excel(writer, 'Puntaje_Z')
#tabla_z_descrip.to_excel(writer, 'Descriptivos_Puntaje Z')
#tabla_temp.to_excel(writer, 'Temperatura')
#tabla_temp_descrip.to_excel(writer, 'Descriptivos_Temperatura')
#writer.save()
Un_día_más = tabla_ori.columns[-1] + datetime.timedelta(days=1)
tablas_ori = pd.concat([tabla_ori, tabla_ori.iloc[:, 1::2], tabla_ori.iloc[:, 2::2]], axis=1)
tablas_ori[Un_día_más] = np.nan
tablas_ori = tablas_ori.sort_index(axis=1)
tablas_unfo = pd.concat([tabla_unfo, tabla_unfo.iloc[:, 1::2], tabla_unfo.iloc[:, 2::2]], axis=1)
tablas_unfo[Un_día_más] = np.nan
tablas_unfo = tablas_unfo.sort_index(axis=1)
tablas_porc = pd.concat([tabla_porc, tabla_porc.iloc[:, 1::2], tabla_porc.iloc[:, 2::2]], axis=1)
tablas_porc[Un_día_más] = np.nan
tablas_porc = tablas_porc.sort_index(axis=1)
tablas_z = pd.concat([tabla_z, tabla_z.iloc[:, 1::2], tabla_z.iloc[:, 2::2]], axis=1)
tablas_z[Un_día_más] = np.nan
tablas_z = tablas_z.sort_index(axis=1)
##### Actograma Datos Originales
tablas_ori.plot.area(subplots=True, layout=(total_dias, 2), color='k', figsize=tamaño, sharex=True, xlim=(tabla_ori.index.min(), tabla_ori.index.max()), ylim=(df['Actividad'].min(), df['Actividad'].max()+2), xticks=horas, yticks=[], rot='vertical', fontsize=15);plt.subplots_adjust(hspace=0);plt.subplots_adjust(wspace=0);plt.xlabel(' ');#plt.xlabel(' ')
plt.savefig('{}/{}_Original_{}.png'.format(directorio, nombre, datetime.datetime.now()), bbox_inches='tight')
plt.show()
##### Actograma Datos Unformat
tablas_unfo.plot.area(subplots=True, layout=(total_dias, 2), color='k', figsize=tamaño, sharex=True, xlim=(tabla_unfo.index.min(), tabla_unfo.index.max()), ylim=(df['Unformat'].min(), df['Unformat'].max()+5), xticks=horas, yticks=[], rot='vertical', fontsize=15);plt.subplots_adjust(hspace=0);plt.subplots_adjust(wspace=0);plt.xlabel(' ');#plt.xlabel(' ')
plt.savefig('{}/{}_Unformat_{}.png'.format(directorio, nombre, datetime.datetime.now()), bbox_inches='tight')
plt.show()
##### Actograma Datos Porcentaje
tablas_porc.plot.area(subplots=True, layout=(total_dias, 2), color='k', figsize=tamaño, sharex=True, xlim=(tabla_porc.index.min(), tabla_porc.index.max()), ylim=(df['Porcentaje'].min(), df['Porcentaje'].max()+0.1), xticks=horas, yticks=[], rot='vertical', fontsize=15);plt.subplots_adjust(hspace=0);plt.subplots_adjust(wspace=0);plt.xlabel(' ');#plt.xlabel(' ')
plt.savefig('{}/{}_Porcentaje_{}.png'.format(directorio, nombre, datetime.datetime.now()), bbox_inches='tight')
plt.show()
##### Actograma Datos Z
tablas_z.plot(subplots=True, layout=(total_dias, 2), color='k', figsize=tamaño, sharex=True, xlim=(tabla_z.index.min(), tabla_z.index.max()), ylim=(df['zscore'].min(), df['zscore'].max()+1), xticks=horas, yticks=[], rot='vertical', fontsize=15);plt.subplots_adjust(hspace=0);plt.subplots_adjust(wspace=0);plt.xlabel(' ');#plt.xlabel(' ')
plt.savefig('{}/{}_Z_{}.png'.format(directorio, nombre, datetime.datetime.now()), bbox_inches='tight')
plt.show()
##########################################
##### Promedio por Días ######
### Tabla Original
tabla_ori.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla_ori.mean(axis=1))+5, 5), label='Promedio', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_Original_Promedio_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()
### Tabla Unformat
tabla_unfo.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla_unfo.mean(axis=1))+50, 50), label='Promedio', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_Unformat_Promedio_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()
### Tabla Porcentaje
tabla_porc.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla_porc.mean(axis=1))+0.05, 0.05), label='Promedio', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_Porcentaje_Promedio_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()
### Tabla Z
tabla_z.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(min(tabla_z.mean(axis=1)), max(tabla_z.mean(axis=1))+0.5, 0.5), label='Promedio', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_Z_Promedio_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()
##### Sumatoria por Días ######
### Tabla Original
tabla_ori.sum(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla_ori.sum(axis=1))+10, 10), label='Sumatoria', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_Original_Sumatoria_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()
### Tabla Unformat
tabla_unfo.sum(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla_unfo.sum(axis=1))+200, 200), label='Sumatoria', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_Unformat_Sumatoria_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()
### Tabla Porcentaje
tabla_porc.sum(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla_porc.sum(axis=1))+0.2, 0.2), label='Sumatoria', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_Porcentaje_Sumatoria_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()
### Tabla Z
tabla_z.sum(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(min(tabla_z.sum(axis=1)), max(tabla_z.sum(axis=1))+2, 2), label='Sumatoria', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_Z_promedio_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()
### Promedio Temperatura °C
tabla_temp.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla_temp.mean(axis=1))+10, 5), label='Prom - Temp °C', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_Temp prom por día_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()
### Sumatoria Temperatura °C
tabla_temp.sum(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla_temp.sum(axis=1))+20, 10), label='Sum - Temp °C', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_Temp sum por día_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()
#################################################################






#plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
#plt.figsize(10, 6)
tabla.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla.mean(axis=1))+1, 0.2), label='Prom - Act', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.subplot(2,1,2)
tabla_temp.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla_temp.mean(axis=1))+10, 10), label='Prom - Temp °C', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_Act - Temp_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()







#writer = pd.ExcelWriter('{}/{} {} {}.xlsx'.format(directorio, serie_acti, nombre, datetime.datetime.now()))
#datos.to_excel(writer, '{}'.format(serie_acti))
#tabla.to_excel(writer, 'Por Hora')
#tabla_descrip.to_excel(writer, 'Descriptivos por día')
#writer.save()




tablas = pd.concat([tabla, tabla.iloc[:, 1::2], tabla.iloc[:, 2::2]], axis=1)
tablas[Un_día_más] = np.nan
tablas = tablas.sort_index(axis=1)
tablas.plot.area(subplots=True, layout=(total_dias, 2), color='k', figsize=tamaño, sharex=True, xlim=(tablas.index.min(), tablas.index.max()), ylim=(0, 50), xticks=horas, yticks=ejey, rot='vertical', fontsize=15); plt.subplots_adjust(hspace=0); plt.subplots_adjust(wspace=0);
plt.savefig('{}/{}_{}.png'.format(directorio, nombre, datetime.datetime.now()), bbox_inches='tight')
plt.show()
##########################################
##### Promedio por días ######
#tabla['Promedio'] = tabla.mean(axis=1)
tabla.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla.mean(axis=1))+20, 20), label='Promedio', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_promedio por día_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()

tabla.sum(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla.sum(axis=1))+20, 20), label='Sumatoria', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_sumatoria por día_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()

### Temperatura °C
tabla_temp.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla.mean(axis=1))+40, 10), label='Temp °C', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_Temp prom por día_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()

tabla_temp.sum(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla.sum(axis=1))+60, 10), label='Temp °C', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_Temp sumatoria por día_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()


tabla.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=horas, yticks=np.arange(0, max(tabla.mean(axis=1))+20, 20), label='Promedio', fontsize=15); plt.xlabel(' '); plt.tight_layout();
tabla_temp.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=ejex, yticks=np.arange(0, max(tabla.mean(axis=1))+40, 10), label='Temp °C', fontsize=15); plt.xlabel(' '); plt.tight_layout();









norte = df['Latitud'][df['Latitud'].str.contains(' *N') == True]
norte = norte.str.strip(' *N')
norte = norte.str.lstrip()
norte = norte.str.rstrip()
#norte = np.array(norte)*-1

sur = df['Latitud'][df['Latitud'].str.contains(' *S') == True]
sur = sur.str.strip(' *S')
sur = sur.str.lstrip()
sur = sur.str.rstrip()
sur = np.multiply(sur, -1)



norte = df['Latitud'].str.contains(' *N')
sur = df['Latitud'].str.contains(' *S')
este = df['Longitude'].str.contains(' *E')
oeste = df['Longitude'].str.contains(' *W')

while True:
    df['Latitud'].str.contains(' *N')
    df['Latitud'].str.strip(' *N')
    df['Latitud'].str.lstrip()
    df['Latitud'].str.rstrip()
    df['Latitud']*1
    break


for True in df['Latitud'].str.contains(' *N'):
    print('Ahuevi')
    if i is True:
        print('aheivo')
        df['Latitud'].str.strip(' *N')
        df['Latitud'].str.lstrip()
        df['Latitud'].str.rstrip().astype(float)*1
        print(df['Latitud'])
    else:
        df['Latitud'].str.strip(' *S')
        df['Latitud'].str.lstrip()
        df['Latitud'].str.rstrip().astype(float)
        df['Latitud']*-1

nort = df['Latitud'][df['Latitud'].str.contains(' *N') == True]
surr = df['Latitud'][df['Latitud'].str.contains(' *S') == True]

        
    
df['Latitud'] = df['Latitud'].str.strip(' *N')
df['Latitud'] = df['Latitud'].str.lstrip()
df['Latitud'] = df['Latitud'].str.rstrip()
df['Latitud']*1
df['Latitud'].str.contains(' *S')
df['Latitud'] = df['Latitud'].str.strip(' *S')
df['Latitud'] = df['Latitud'].str.lstrip()
df['Latitud'] = df['Latitud'].str.rstrip().astype(float)
df['Latitud']*-1

df['Latitud'] = df['Latitud'].str.strip(' *N')
df['Latitud'] = df['Latitud'].str.lstrip()
df['Latitud'] = df['Latitud'].str.rstrip().astype(float)*1
df['Latitud'] = df['Latitud'].str.strip(' *S')
df['Latitud'] = df['Latitud'].str.lstrip()
df['Latitud'] = df['Latitud'].str.rstrip().astype(float)*-1



df['Latitud'].str.contains(' *N')
df['Latitud'].str.contains(' *S')

df['Latitud'] = df['Latitud'].str.strip(' *N')
df['Latitud'] = df['Latitud'].str.lstrip()
df['Latitud'] = df['Latitud'].str.rstrip().astype(float)




#df = df.iloc[4: , :]
nombre = '{:.10}'.format(nombre)
fecha = df['Date']
hora = df['Time']
intervalo = fecha + ' ' + hora
intervalo = pd.to_datetime(intervalo)


salida_sol = '06:00:00'
puesta_sol = '18:00:00'
#intervalo_inicial = fecha + ' ' + '00:00:00'
frecu = '5 min'
if frecu == '1 min':
    frecu = '1T'
    len_datos = 1440
elif frecu == '5 min':
    frecu = '5T'
    len_datos = 288
edad = ' '
serie_acti = dfnom[5:9]
sexo = 'M'
if sexo == 'M':
    sexo = 'Masculino'
elif sexo == 'F':
    sexo = 'Femenino'

df['Altitude'] = df['Altitude'].str.strip(' mt')
df['Altitude'] = df['Altitude'].str.lstrip()
df['Altitude'] = df['Altitude'].str.rstrip().astype(float)
df['Latitude'] = df['Latitude'].str.strip(' *N')
df['Latitude'] = df['Latitude'].str.lstrip()
df['Latitude'] = df['Latitude'].str.rstrip().astype(float)
df['Longitude'] = df['Longitude'].str.strip(' *W')
df['Longitude'] = df['Longitude'].str.lstrip()
df['Longitude'] = df['Longitude'].str.rstrip().astype(float)
df['Longitude'] = df['Longitude']*-1
df['Speed'] = df['Speed'].str.strip(' m/s')
df['Speed'] = df['Speed'].str.lstrip()
df['Speed'] = df['Speed'].str.rstrip().astype(float)
#df['Activity Level'] = (df['Activity Level']*999)/df['Activity Level'].max()
df['Temperature'] = df['Temperature'].str.strip(' oC')
df['Temperature'] = df['Temperature'].str.lstrip()
df['Temperature'] = df['Temperature'].str.rstrip().astype(float)
df.to_csv('{}/{}_modificado.csv'.format(directorio, nombre), encoding='UTF-8')

df = df.set_index(intervalo)
sum_dias = round(df.loc[:, 'Activity Level'].resample('1D').mean()).min()*100
total_dias = len(df.resample('1D'))
if total_dias in np.arange(2, 31, 1):
    tamaño = (8, 15)
    ejex = ['00:00:00', '06:00:00', '12:00:00', '18:00:00', '23:59:00']
    ejey = []
    etiquetax = ''
    titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
    tamaño_letra = 15
    etiquetay = [0, 500, 1000]
    etiquetayprom = np.arange(0, 1100, 100)
    etiquetaysum = np.arange(0, sum_dias, 100)
    local_flecha = 10
    local_texto = -100
elif total_dias in np.arange(31, 91, 1):
    tamaño = (10, 40)
    ejex = ['00:00:00', '06:00:00', '12:00:00', '18:00:00', '23:59:00']
    ejey = []
    etiquetax = ''
    titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
    tamaño_letra = 20
    etiquetay = [0, 2500]
    etiquetayprom = np.arange(0, 1100, 100)
    etiquetaysum = np.arange(0, sum_dias, 100)
    local_flecha = 5
    local_texto = -30
elif total_dias in np.arange(91, 182, 1):
    tamaño = (10, 50)
    ejex = ['00:00:00', '06:00:00', '12:00:00', '18:00:00', '23:59:00']
    ejey = []
    etiquetax = ''
    titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
    tamaño_letra = 25
    etiquetay = [0, 2500]
    etiquetayprom = np.arange(0, 1100, 100)
    etiquetaysum = np.arange(0, sum_dias, 100)
    local_flecha = 10
    local_texto = -100
elif total_dias in np.arange(182, 366, 1):
    tamaño = (12, 180)
    ejex = ['00:00:00', '06:00:00', '12:00:00', '18:00:00', '23:59:00']
    ejey = []
    etiquetax = ''
    titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
    tamaño_letra = 25
    etiquetay = [0, 1000]
    etiquetayprom = np.arange(0, 1100, 100)
    etiquetaysum = np.arange(0, sum_dias, 100)
    local_flecha = 300
    local_texto = 100
df['Date'] = pd.to_datetime(df['Date']).dt.date.to_frame()
df['Time'] = pd.to_datetime(df['Time']).dt.time.to_frame()
tabla = df.pivot_table(values='Activity Level', index='Time', columns='Date')
tabla_temp = df.pivot_table(values='Temperature', index='Time', columns='Date')
Un_día_más = tabla.columns[-1] + datetime.timedelta(days=1)
tabla1 = tabla.iloc[:, 1:]
tabla1[Un_día_más] = np.nan
tabla_descrip = tabla.describe()
tablas = pd.concat([tabla, tabla1], axis=1, keys=['Act1', 'Act2'], names=['Actogramas', 'Fecha'])
tablas.info()
tablas.columns.names[0]
nombre_act1 = tablas.columns.levels[0][0]
nombre_act2 = tablas.columns.levels[0][1]
act1 = tablas.xs('Act1', axis=1)
act2 = tablas.xs('Act2', axis=1)
tablas.xs('Act1', axis=1).plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(min(act1.sum(axis=1)), max(act1.sum(axis=1))), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.xlabel(titulo, fontsize=tamaño_letra); plt.subplots_adjust(hspace=0);
#plt.savefig('{}/{}_{}_{}.png'.format(directorio, nombre, nombre_act1, datetime.datetime.now()))
tablas.xs('Act2', axis=1).plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(min(act2.sum(axis=1)), max(act2.sum(axis=1))), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.xlabel(titulo, fontsize=tamaño_letra); plt.subplots_adjust(hspace=0);
#plt.savefig('{}/{}_{}_{}.png'.format(directorio, nombre, nombre_act2, datetime.datetime.now()))
#writer = pd.ExcelWriter('{}/{}_{}.xlsx'.format(directorio, serie_acti, nombre))
#datos.to_excel(writer, '{}'.format(serie_acti))
#tabla.to_excel(writer, 'Por Hora')
#tabla_descrip.to_excel(writer, 'Descriptivos por día')
#writer.save()
#######################################################
##### GRAFICAR MAPAS 
mexico = df.loc[:, ['Latitude', 'Longitude']]
mi_mapa = folium.Map(location=(19.4284700, -99.1276600), zoom_start=13)
marcador1 = folium.Marker(location=(mexico.iloc[0, 0], mexico.iloc[0, 1]))
marcador2 = folium.Marker(location=(mexico.iloc[1, 0], mexico.iloc[1, 1]))
marcador3 = folium.Marker(location=(mexico.iloc[2, 0], mexico.iloc[2, 1]))
marcador4 = folium.Marker(location=(mexico.iloc[3, 0], mexico.iloc[3, 1]))
marcador1.add_to(mi_mapa)
marcador2.add_to(mi_mapa)
marcador3.add_to(mi_mapa)
marcador4.add_to(mi_mapa)
mi_mapa.save("mapa.html")

continents = Continent.filter()
for cont in continents:
    print(cont.name)

latitude = "19.4284700"
longitude = "-99.1276600"
lat_long = "{0},{1}".format(latitude, longitude)

finder = PlaceFinder.get(text=lat_long, gflags="R")
place_info = "{0}, {1} ZIP: {2} | WOEID:{3}".format(finder.city,
                                                    finder.country,
                                                    finder.uzip,
                                                    finder.woeid)
print(place_info)
###########################################################
##### Actograma 1 #####
act1.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(act1.index.min(), act1.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.xlabel(titulo, fontsize=tamaño_letra); plt.subplots_adjust(hspace=0);
plt.savefig('{}/{}_{}_{}.png'.format(directorio, nombre, nombre_act1, datetime.datetime.now()))
plt.show()
##### Actograma 2 #####
act2.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(act2.index.min(), act2.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.xlabel(titulo, fontsize=tamaño_letra); plt.subplots_adjust(hspace=0);
plt.savefig('{}/{}_{}_{}.png'.format(directorio, nombre, nombre_act2, datetime.datetime.now()))
plt.show()
##########################################################
fig, axes = plt.subplots(nrows=2, ncols=2)
act1.plot.area(ax=axes[0,0], subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.subplots_adjust(hspace=0); 
act2.plot.area(ax=axes[0,1], subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.subplots_adjust(hspace=0);

#plt.subplots_adjust(hspace=0)
#plt.subplots_adjust(wspace=0)
plt.subplot2grid((1,2), (0,0)); 
act1.plot.area(subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.xlabel(titulo, fontsize=tamaño_letra); plt.subplots_adjust(hspace=0); 
plt.subplot2grid((1,2), (0,1)); 
act2.plot.area(subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.xlabel(titulo, fontsize=tamaño_letra); plt.subplots_adjust(hspace=0);


act1.plot.area(subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.xlabel(titulo, fontsize=tamaño_letra); plt.subplots_adjust(hspace=0);
act2.plot.area(subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.xlabel(titulo, fontsize=tamaño_letra); plt.subplots_adjust(hspace=0);
#fig.subplots_adjust(hspace=0)   
#fig.tight_layout()
plt.savefig('{}/Ejemplo-2.png'.format(directorio))

plt.figure()
plt.subplot(1, 2, 1)
act1.plot.area(subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.xlabel(titulo, fontsize=tamaño_letra); plt.subplots_adjust(hspace=0);
plt.subplot(1, 2, 2)
act2.plot.area(subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15); plt.xlabel(titulo, fontsize=tamaño_letra); plt.subplots_adjust(hspace=0);
plt.subplots_adjust(wspace=0)

####################################################
### PERIODICIDAD ######
amplitud = 3
periodo = np.pi


t = np.arange(-1, 10, 0.001)
funcion = ((sp.square(2 * t)) * (amplitud / 2.0)) + (amplitud / 2.0)

plt.plot(t, funcion, lw=2)
plt.grid()
plt.annotate('Pi', xy = (np.pi, 1), xytext = (np.pi, 1.1))
plt.annotate('Pi/2', xy = (np.pi / 2.0, 1), xytext = (np.pi / 2.0, 1.1))
plt.ylabel('Amplitud')
plt.xlabel('Tiempo(t)')
plt.ylim(-amplitud, amplitud + 1)
plt.xlim(-0.5, 4)
plt.show()

##########################################
##### Promedio por días ######
#tabla['Promedio'] = tabla.mean(axis=1)
act1.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=ejex, yticks=np.arange(0, max(act1.sum(axis=1))+3), label='Promedio', fontsize=15); plt.xlabel(' '); plt.tight_layout();
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_promedio por día_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()

act1.sum(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=ejex, yticks=np.arange(0, max(act1.sum(axis=1))+3), label='Sumatoria', fontsize=15); plt.xlabel(' ')
plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_sumatoria por día_{}.png'.format(directorio, nombre, datetime.datetime.now()))
plt.show()

#tabla.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=ejex, yticks=ejey, label='media', fontsize=15)
#tabla.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=ejex, yticks=ejey, label='media', fontsize=15)
#tabla.sum(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=ejex, yticks=ejey, label='media', fontsize=15)
#plt.axvline(salida_sol, color='b', label='Mañana', linestyle='--')
#plt.axvline(puesta_sol, color='r', label='Noche', linestyle='-.')
#plt.axhline(act1.quantile(.25).max(), color='g', label='25%', linestyle=':')
#plt.axhline(act1.quantile(.5).max(), color='c', label='50%', linestyle=':')
#plt.axhline(act1.quantile(.75).max(), color='m', label='75%', linestyle=':')

plt.subplot(2, 1, 1)
act1.sum(axis=1).plot(title=nombre, color='k', figsize=(10, 6), xticks=ejex, yticks=np.arange(0, max(act1.sum(axis=1))+3), label='Sumatoria', fontsize=15); plt.xlabel(' ')
plt.yticks(fontsize=15)
plt.legend(loc='best', fontsize=15)
plt.subplot(2, 1, 2)
tabla_temp.mean(axis=1).plot(title=nombre, color='k', figsize=(20, 16), xticks=ejex, yticks=np.arange(0, 50, 10), label='Temperatura °C - Promedio', fontsize=20); plt.xlabel(' ')
plt.yticks(fontsize=15)
#plt.axhline(tabla_temp.mean()[0], color='r', label='Temp-Promedio-{}'.format(tabla_temp.columns[0]), linestyle=':')
#plt.axhline(tabla_temp.mean()[1], color='g', label='Temp-Promedio-{}'.format(tabla_temp.columns[1]), linestyle=':')
#plt.axhline(tabla_temp.mean()[2], color='b', label='Temp-Promedio-{}'.format(tabla_temp.columns[2]), linestyle=':')
plt.legend(loc='best', fontsize=15)
plt.tight_layout()
plt.savefig('{}/{}_Act-Temp_{}.png'.format(directorio, nombre, datetime.datetime.now()))

tabla_temp.mean()[0]
tabla_temp.columns[0]
tabla_temp.mean()[1]
tabla_temp.mean()[2]


##### Reacomodar el Actograma #####
dias_inicio_quitados = 20
#dias_ini = int(input('¿Cuántos días habra que quitar al inicio? -> '))
dias_final_quitados = 20
#dias_qui = int(input('¿Cuántos días habra que quitar al final? -> '))
tabla_filtrada = tabla.iloc[:, dias_inicio_quitados:-dias_final_quitados]
tabla_filtrada.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000))
plt.xlabel(titulo)#, fontsize=tamaño_letra)
plt.xticks(ejex)#, fontsize=20)
plt.yticks(etiquetay, ['', ''])
plt.tight_layout()
plt.savefig('{}/{}_filtrada.png'.format(directorio, nombre))
plt.show()
len(tabla_filtrada.columns)
##### Promedio por días ######
tabla_filtrada.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6))
plt.xlabel(etiquetax)
plt.xticks(ejex, fontsize=15)
plt.axvline(salida_sol, color='b', label='Mañana', linestyle='--')
plt.axvline(puesta_sol, color='r', label='Noche', linestyle='-.')
plt.tight_layout()
plt.legend(loc='best')
plt.savefig('{}/{}_filtrada_promedio por día.png'.format(directorio, nombre))
plt.show()

##################################################################
#### Actograma realizandose
tablas = pd.concat([tabla, tabla1], axis=1, keys=['Act1', 'Act2'], names=['Actogramas', 'Fecha'])
tablas.info()
tablas.columns.names[0]
act1 = tablas.xs('Act1', axis=1)
act2 = tablas.xs('Act2', axis=1)
tablas.xs('Act1', axis=1).plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
tablas.xs('Act2', axis=1).plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
tablas.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
#################################################
tablas.iloc[:, :total_dias]
plt.subplot(1,2,1)
plt.fill(tablas.iloc[:, :total_dias], 'k')
plt.subplot(1,2,2)
plt.fill(tablas.iloc[:, total_dias:], 'k');

tablas['Act1']
tablas['Act1'].values
tablas.loc[:, slice('Act1')]
tablas1 = tablas.groupby(axis=1, level='Actogramas')

tablas1.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)


for acts, group in tablas1:
   # print(acts)
    #print(group)
    #print(tablas1.get_group(acts))
    plt.figure()
    plt.fill(acts, group)



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=tamaño)
tablas.groupby(axis=1, level='Actogramas').plot()

for i in range(tablas.shape[1]):
    print(i[0:13])
    plt.subplot(1, 1, i)
    plt.fill_between(i, 'k')
    

x=[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
y=[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
for i in range(len(x)):
    print(i)
    plt.figure()
    plt.plot(x[i],y[i])
    



###########################################
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
act1.plot.area(subplots=True, sharex=True, color='k', xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
ax2 = fig.add_subplot(1, 2, 2)
act2.plot.area(ax=ax2, subplots=True, sharex=True, color='k', xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=tamaño)
plt.subplots_adjust(wspace=0)
act1.plot.area(ax=axes[0], subplots=True, sharex=True, color='k', xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
act2.plot.area(ax=axes[1], subplots=True, sharex=True, color='k', xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)


fig, axes = plt.subplots(figsize=tamaño)
gs = gridspec.GridSpec(1, 2) 
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharey=ax1)
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
plt.subplots_adjust(wspace=0)
act1.plot.area(ax=ax1, subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
act2.plot.area(ax=ax2, subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)





fig = plt.figure(figsize=tamaño)
plt.subplots_adjust(hspace=0)
plt.subplots_adjust(wspace=0)
ax1 = plt.subplot2grid((1,2), (0,0))
ax2 = plt.subplot2grid((1,2), (0,1))
#act1.plot.area(ax=ax1, subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
#act2.plot.area(ax=ax2, subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
#fig.subplots_adjust(hspace=0)   
#fig.tight_layout()
plt.savefig('{}/Ejemplo-1.png'.format(directorio))




plt.figure(1)
plt.subplot(1, 2, 1)
act1.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
plt.subplot(1, 2, 2)
act2.plot.area(ax=plt.gca(), subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
plt.tight_layout()




f, ax = plt.subplots(1,2, figsize=tamaño)
plt.figure()
act1.plot.area(stacked=True, ax=ax[0], subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
plt.figure()
act2.plot.area(stacked=True, ax=ax[1], subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)


fig = plt.figure()
ax1 = plt.subplot2grid((1,2), (0,0))
ax2 = plt.subplot2grid((1,2), (0,1))
act1.plot.area(ax=ax1, subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
act2.plot.area(ax=ax2, subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)


fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=tamaño)
act1.plot.area(ax=axes[0], subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
act2.plot.area(ax=axes[1], subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)


fig, axes = plt.subplots(1, 2, figsize=tamaño)
plt.figure()
tablas.xs('Act1', axis=1).plot.area(subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15, ax=axes[0])
plt.figure()
tablas.xs('Act2', axis=1).plot.area(subplots=True, color='b', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15, ax=axes[1])



fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, sharey=True, figsize=tamaño)
tablas.xs('Act1', axis=1).plot.area(ax=ax0, subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
tablas.xs('Act2', axis=1).plot.area(ax=ax1, subplots=True, color='r', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)

tablas.swaplevel(axis=1).info()
tablas.unstack(level=1).plot(kind='area', subplots=True)



tablas1 = tablas.T
tablas2 = tablas1.groupby(level='Actogramas')
tablas2.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
tablas2.describe()
tablas2.plot.area(subplots=True, color='k', figsize=tamaño)


tablas.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)

tablas.columns

axes = tabla.plot(subplots=True, color='k')
for i, ax in enumerate(axes):
    tabla1[i].plot(ax=ax)
plt.draw()

axes = tabla.plot(subplots=True, color='k')
for i, ax in enumerate(axes):
    tabla1.iloc[:, i].plot(ax=ax)
plt.draw()

# Get the figure and the axes
#(np.array(tamaño)*2)
fig, axes = plt.subplots(1, 2, figsize=tamaño)
tabla.plot.area(subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15, ax=axes[0])
tabla1.plot.area(subplots=True, color='b', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15, ax=axes[1])

fig = plt.figure(figsize=tamaño)
sub1 = fig.add_subplot(121)
tabla.plot.area(subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15, ax=sub1)
sub2 = fig.add_subplot(122)
tabla1.plot.area(subplots=True, color='b', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15, ax=sub2)
plt.tight_layout()


fig, [ax0, ax1] = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=tamaño)
tabla.plot.area(subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15, ax=ax0)
tabla1.plot.area(subplots=True, color='b', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15, ax=ax1)

ax0.set_xlim(tabla.index.min(), tabla.index.max())
ax0.set_ylim(0, 1000)

tabla.iloc[:, 1:].plot.area(subplots=True, color='b', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15, ax=ax1)
ax1.set_xlim(tabla.index.min(), tabla.index.max())
ax1.set_ylim(0, 1000)

fig.tight_layout()
plt.savefig('{}/Ejemplo-1.png'.format(directorio))



##########################################


act1.plot.area(ax=ax0, subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
act2.plot.area(ax=ax1, subplots=True, color='k', sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)


fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, sharey=True, figsize=(7, 4))
top_10.plot(kind='barh', y="Sales", x="Name", ax=ax0)
ax0.set_xlim([-10000, 140000])
ax0.set(title='Revenue', xlabel='Total Revenue', ylabel='Customers')

# Plot the average as a vertical line
avg = top_10['Sales'].mean()
ax0.axvline(x=avg, color='b', label='Average', linestyle='--', linewidth=1)

# Repeat for the unit plot
top_10.plot(kind='barh', y="Purchases", x="Name", ax=ax1)
avg = top_10['Purchases'].mean()
ax1.set(title='Units', xlabel='Total Units', ylabel='')
ax1.axvline(x=avg, color='b', label='Average', linestyle='--', linewidth=1)

# Title the figure
fig.suptitle('2014 Sales Analysis', fontsize=14, fontweight='bold');

# Hide the legends
ax1.legend().set_visible(False)
ax0.legend().set_visible(False)



########################################
fig = plt.figure(figsize=tamaño)

ax1 = plt.subplot2grid((1,2), (0,0))
ax2 = plt.subplot2grid((1,2), (0,1))
act1.plot.area(ax=ax1, subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
act2.plot.area(ax=ax2, subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)

fig.subplots_adjust(hspace=0)   

fig.tight_layout()
plt.savefig('{}/Ejemplo-1.png'.format(directorio))

##########################################
t = np.arange(0.0, 2.0, 0.01)

s1 = np.sin(2*np.pi*t)
s2 = np.exp(-t)
s3 = s1*s2

fig = plt.figure()
ax1 = plt.subplot2grid((4,3), (0,0), colspan=3, rowspan=2)
ax2 = plt.subplot2grid((4,3), (2,0), colspan=3, sharex=ax1)
ax3 = plt.subplot2grid((4,3), (3,0), colspan=3, sharex=ax1)

ax1.plot(t,s1)
ax2.plot(t[:150],s2[:150])
ax3.plot(t[30:],s3[30:])

fig.subplots_adjust(hspace=0)   
for ax in [ax1, ax2]:
    plt.setp(ax.get_xticklabels(), visible=False)
    # The y-ticks will overlap with "hspace=0", so we'll hide the bottom tick
    ax.set_yticks(ax.get_yticks()[1:])  
fig.tight_layout()
plt.show()
###########################################
dist_norm = np.random.normal(loc=0, scale=1, size=1000)
dist_tdis = np.random.standard_t(df=29, size=1000)
dist_fdis = np.random.f(dfnum=59, dfden=28, size=1000)
dist_chsq = np.random.chisquare(df=2, size=1000)

# Plot figure with subplots of different sizes
fig = plt.figure(1)
# set up subplot grid
gridspec.GridSpec(3,3)

# large subplot
plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.title('Normal distribution')
plt.xlabel('Data values')
plt.ylabel('Frequency')
plt.hist(dist_norm, bins=30, color='0.30')

# small subplot 1
plt.subplot2grid((3,3), (0,2))
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.title('t distribution')
plt.xlabel('Data values')
plt.ylabel('Frequency')
plt.hist(dist_tdis, bins=30, color='b')

# small subplot 2
plt.subplot2grid((3,3), (1,2))
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.title('F distribution')
plt.xlabel('Data values')
plt.ylabel('Frequency')
plt.hist(dist_fdis, bins=30, color='r')

# small subplot 3
plt.subplot2grid((3,3), (2,2))
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.title('Chi-square distribution')
plt.xlabel('Data values')
plt.ylabel('Frequency')
plt.hist(dist_chsq, bins=30, color='g')

# fit subplots and save fig
fig.tight_layout()
fig.set_size_inches(w=11,h=7)
#fig_name = 'plot.png'
#fig.savefig(fig_name)
plt.savefig('{}/Ejemplo.png'.format(directorio))



################################################
gs = gridspec.GridSpec(1, 2)
ax1 = plt.subplot(gs[tabla.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)])
ax2 = plt.subplotgs(gs[tabla1.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)])


#############################################
fig, axes = plt.subplots(1, 2, sharey=True)
plt.subplots_adjust(wspace=0, hspace=0)
axes[0] = tabla.plot.area(subplots=True, color='k', figsize=tamaño, xlim=(tabla.index.min(), tabla.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
axes[1] = tabla1.plot.area(subplots=True, color='k', figsize=tamaño, xlim=(tabla1.index.min(), tabla1.index.max()), ylim=(0, 1000), xticks=ejex, yticks=ejey, rot='vertical', fontsize=15)
plt.tight_layout()

##############################################
plt.figure(figsize=(8, 15), dpi=80)
ax1 = plt.subplot(1, 2, 1) 
ax1 = plt.fill_between(tabla.index, tabla.iloc[:, 0], color='k')
plt.xlim(tabla.index.min(), tabla.index.max())
plt.xticks(ejex)
plt.ylim(0, 1000)
ax2 = plt.subplot(1, 2, 2) 
ax2 = plt.fill_between(tabla.index, tabla.iloc[:, 1], color='k')
plt.xlim(tabla.index.min(), tabla.index.max())
plt.xticks(ejex)
plt.ylim(0, 1000)
##############################################
df = pd.DataFrame(np.random.rand(700,6))

col_per_plot = 3
cols = df.columns.tolist()
# Create groups of 3 columns
cols_splits = [cols[i:i+col_per_plot] for i in range(0, len(cols), col_per_plot)]  

# Define plot grid.
# Here I assume it is always one row and many columns. You could fancier...
fig, axarr = plt.subplots(1, len(cols_splits))
# Plot each "slice" of the dataframe in a different subplot
for cc, ax in zip(cols_splits, axarr):
    df.loc[:, cc].plot(ax = ax)
################################################
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)

fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
axes[0, 0].plot(x, y)
axes[1, 1].scatter(x, y)
plt.tight_layout()

plt.subplots(2, 2, sharex='col')

plt.subplots(2, 2, sharey='row')

plt.subplots(2, 2, sharex='all', sharey='all')

plt.subplots(2, 2, sharex=True, sharey=True)
#############################################################












#### Actograma con seaborn ####
tabla.axes
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
for tabla.index, tabla.columns in tabla:
    print(tabla.index)
    print(tabla.columns)
    plt.figure()
    plt.fill_between(x[i], y[i], color='k')

plt.figure(figsize=(8, 15), dpi=80)
ax1 = plt.subplot(2, 1, 1) 
ax1 = plt.fill_between(tabla.index, tabla.iloc[:, 0], color='k')
plt.xlim(tabla.index.min(), tabla.index.max())
plt.xticks(ejex)
plt.ylim(0, 1000)
ax2 = plt.subplot(2, 1, 2) 
ax2 = plt.fill_between(tabla.index, tabla.iloc[:, 1], color='k')
plt.xlim(tabla.index.min(), tabla.index.max())
plt.xticks(ejex)
plt.ylim(0, 1000)

## Ejemplo 1 ##
fig, axs = plt.subplots(1, 2)
for tick in axs[0].get_xticklabels():
    tick.set_rotation(55)
axs[0].set_xlabel('XLabel 0')
axs[1].set_xlabel('XLabel 1')
fig.align_xlabels()

#######################################################
plt.figure(figsize=tamaño, dpi=80)
ax1 = plt.subplot(1, 2, 1)
ax1 = tabla.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, sharey=True)
plt.xlabel(titulo)#, fontsize=tamaño_letra)
plt.xticks(ejex)#, fontsize=20)
plt.yticks(etiquetay, ['', ''])
plt.tight_layout()
ax2 = plt.subplot(1, 2, 2)
ax2 = tabla1.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, sharey=True)
plt.xlabel(titulo)#, fontsize=tamaño_letra)
plt.xticks(ejex)#, fontsize=20)
plt.yticks(etiquetay, ['', ''])
plt.tight_layout()


##### Prueba Actograma #####
gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1,:-1])
ax3 = plt.subplot(gs[1:, -1])
ax4 = plt.subplot(gs[-1, 0])
ax5 = plt.subplot(gs[-1, -2])

gs1 = gridspec.GridSpec(3, 3)
gs1.update(left=0.05, right=0.48, wspace=0.05)
ax1 = plt.subplot(gs1[:-1, :])
ax2 = plt.subplot(gs1[-1, :-1])
ax3 = plt.subplot(gs1[-1, -1])
gs2 = gridspec.GridSpec(3, 3)
gs2.update(left=0.55, right=0.98, hspace=0.05)
ax4 = plt.subplot(gs2[:, :-1])
ax5 = plt.subplot(gs2[:-1, -1])
ax6 = plt.subplot(gs2[-1, -1])

gs1 = gridspec.GridSpec(1, 2)
#gs1.update(left=0.05, right=0.48, wspace=0.05)
ax1 = plt.subplot(gs1[0, 1])
ax1 = tabla.plot.area(subplots=True, color='k', figsize=tamaño)
gs2 = gridspec.GridSpec(1, 2)
#gs2.update(left=0.55, right=0.98, hspace=0.05)
ax2 = plt.subplot(gs2[:, :])
ax2 = tabla1.plot.area(subplots=True, color='k', figsize=tamaño)

gs0 = gridspec.GridSpec(1, 2)
gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])
gs01 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[1])

gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1,2],
                       height_ratios=[4,1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])
help(gridspec.GridSpec)

plt.figure(figsize=(16, 30))
G = gridspec.GridSpec(1, 2)
axes_1 = plt.subplot(G[0, :])
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'Axes 1', ha='center', va='center', size=24, alpha=.5)
axes_3 = plt.subplot(G[1:, -1])
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'Axes 3', ha='center', va='center', size=24, alpha=.5)


plt.figure(figsize=(8, 15))
G = gridspec.GridSpec(1, 2)
axes_1 = plt.subplot(G[0])
plt.xticks(())
plt.yticks(())
axes_1 = tabla.plot.area(subplots=True, color='k', figsize=tamaño)
axes_2 = plt.subplot(G[-1])
plt.xticks(())
plt.yticks(())
axes_2 = tabla1.plot.area(subplots=True, color='k', figsize=tamaño)
plt.tight_layout()


fig, axes = plt.subplots(ncols=2, sharey=True)
plt.setp(axes, title='Test')
fig.suptitle('An overall title', size=20)
fig.tight_layout()
fig.subplots_adjust(top=0.9) 
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(ncols=2, sharey=True)
plt.setp(axes, title='Actividad')
#fig.suptitle(titulo)
axes1 = tabla.plot.area(subplots=True, color='k', figsize=tamaño, ax=axes[0])
axes2 = tabla1.plot.area(subplots=True, color='k', figsize=tamaño, ax=axes[-1])
fig.tight_layout()
fig.subplots_adjust(top=1) 
plt.tight_layout()
plt.show()

fig = plt.figure()
gs = gridspec.GridSpec(1,2)
ax1 = fig.add_subplot(gs[0])
ax1 = tabla.plot.area(subplots=True, color='k', figsize=tamaño)
ax2 = fig.add_subplot(gs[1], sharey=ax1)
ax2 = tabla1.plot.area(subplots=True, color='k', figsize=tamaño)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp([ax1, ax2], title='Test')
fig.suptitle('An overall title', size=20)
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
plt.show()




gs1 = gridspec.GridSpec(3, 3)
gs1.update(left=0.05, right=0.48, wspace=0.05)
ax1 = plt.subplot(gs1[:-1, :])
ax2 = plt.subplot(gs1[-1, :-1])
ax3 = plt.subplot(gs1[-1, -1])
plt.tight_layout()
gs2 = gridspec.GridSpec(3, 3)
gs2.update(left=0.55, right=0.98, hspace=0.05)
ax4 = plt.subplot(gs2[:, :-1])
ax5 = plt.subplot(gs2[:-1, -1])
ax6 = plt.subplot(gs2[-1, -1])
plt.tight_layout()


ax1 = plt.subplot(1, 2, 1)
tabla.plot.area(ax=ax1, subplots=True, color='k', figsize=tamaño)
ax2 = plt.subplot(1, 2, 2)
tabla1.plot.area(ax=ax2, subplots=True, color='k', figsize=tamaño)



fig, axes = plt.subplots(nrows=1, ncols=2)
tabla.plot(kind='area', ax=axes[0,0], subplots=True, color='k', figsize=tamaño)
tabla1.plot(kind='area', ax=axes[0,1], subplots=True, color='k', figsize=tamaño)


fig = plt.figure()
ax1 = tabla.plot.area(subplots=True, color='k', figsize=tamaño)
tabla1.plot.area(subplots=True, color='k', figsize=tamaño)

#fig = plt.figure()
fig = plt.subplots(nrows=1, ncols=2)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(tabla.plot.area(subplots=True, color='k'))
ax2.plot(tabla1.plot.area(subplots=True, color='k'))


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.tabla.plot.area(subplots=True, color='k')
ax2.tabla1.plot.area(subplots=True, color='k')

plt.xlabel(titulo)#, fontsize=tamaño_letra)
plt.xticks(ejex)#, fontsize=20)
plt.yticks(etiquetay, ['', ''])
plt.tight_layout()
plt.savefig('{}/{}.png'.format(directorio, nombre))
plt.show()
############################################

fig, axes = plt.subplots(nrows=1, ncols=2)
tabla.plot.area(subplots=True, color='k', figsize=tamaño, ax=axes[0])
tabla1.plot.area(subplots=True, color='k', figsize=tamaño, ax=axes[1])

fig1 = plt.figure()
fig2 = plt.figure()
ax1 = fig1.add_subplot(1, 2, 1)
ax2 = fig2.add_subplot(1, 2, 2)
tabla.plot.area(subplots=True, color='k', figsize=tamaño, ax=ax1)
tabla1.plot.area(subplots=True, color='k', figsize=tamaño, ax=ax2)


fig, axes = plt.subplots(nrows=1, ncols=2)
tabla.plot.area(subplots=True, color='k', figsize=tamaño, ax=axes[0,0])
tabla1.plot.area(subplots=True, color='k', figsize=tamaño, ax=axes[0,1])


x = [1,2,3,4,5]
y = [5,4,3,2,1]
z = [1,3,5,7,9]
fig1 = plt.figure()
fig2 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
plt.plot(x)
ax2 = fig2.add_subplot(2, 1, 1)
plt.plot(y)
ax3 = fig2.add_subplot(2, 1, 2)
plt.plot(z)



fig1 = plt.figure()
fig2 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax2 = fig2.add_subplot(1, 2, 1)
ax3 = fig2.add_subplot(1, 2, 2)

# Pegar gráficas
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.8])
ax2 = fig.add_axes([0.5, 0.1, 0.4, 0.8])

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1 = tabla.plot.area(subplots=True, color='k', figsize=tamaño)
ax2 = fig.add_subplot(122)
ax2 = tabla1.plot.area(subplots=True, color='k', figsize=tamaño)

f, (ax1, ax2) = plt.subplots(1, 2, sharex='col')
ax1.plot(tabla)
ax2.plot(tabla1)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
tabla.plot.area(subplots=True, color='k', figsize=tamaño, ax=ax1)
tabla1 = tabla.iloc[:, 1:]
tabla1['Un_dia_mas'] = np.nan
tabla1.plot.area(subplots=True, color='k', figsize=tamaño, ax=ax2)

fig, axes = plt.subplots(nrows=1, ncols=2)
tabla.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, ax=axes[0])
tabla1.plot.area(subplots=True, color='k', figsize=tamaño, sharex=True, ax=axes[1])

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)


#####################################################
ax = tabla2.plot.area(subplots=True, color='k')
tabla.plot.area(subplots=True, color='k', ax=ax)

ax1 = tabla.plot.area(subplots=True, color='k')
ax2 = tabla2.plot.area(subplots=True, color='k')

plt.subplot(1, 2, 1)
ax1
plt.subplot(1, 2, 2)
ax2
plt.show()

plt.subplot(2, 1, 1)
plt.subplot(2, 3, 4)
plt.subplot(2, 3, 5)
plt.subplot(2, 3, 6)



tabla2 = tabla.iloc[:, 1:]
tabla2.columns
tabla.columns
sns.kdeplot(tabla, shade=True, color="r")

t = np.arange(0.0, 20.0, 1)
s = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.subplot(1,2,1)
plt.xticks([]), plt.yticks([])
plt.title('subplot(1, 2, 1)')
plt.plot(t,s)
plt.subplot(1,2,2)
plt.xticks([]), plt.yticks([])
plt.title('subplot(1, 2, 2)')
plt.plot(t,s,'r-')
plt.show()


fig = plt.figure()
fig.add_subplot(221)   #top left
fig.add_subplot(222)   #top right
fig.add_subplot(223)   #bottom left
fig.add_subplot(224)   #bottom right 
plt.show()

fig = plt.figure()
fig.add_subplot(1, 2, 1)   #top and bottom left
fig.add_subplot(2, 2, 2)   #top right
fig.add_subplot(2, 2, 4)   #bottom right 
plt.show()

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
fig = plt.figure()
fig.add_subplot(111)
plt.scatter(x, y)
plt.show()












plt.figure(dpi=80)
plt.subplot(1, 2, 1)
tabla.plot(kind='area', subplots=True, color='k', figsize=(25, 33))
plt.axvline(salida_sol, color='r')
plt.xticks(fontsize=30)
plt.subplot(1, 2, 2)
tabla.plot(kind='area', subplots=True, color='k', figsize=(25, 33))
plt.xticks(fontsize=30)
plt.show()


plt.figure(dpi=80)
plt.subplot(1, 2, 1)
sns.kdeplot(tabla, shade=True, color='k')
plt.subplot(1, 2, 2)
sns.kdeplot(tabla.iloc[:, 1:], shade=True, color='k')

tabla.plot.area(subplots=True, color='k', figsize=(25, 33))
plt.subplot(1, 2, 2)
tabla.iloc[:, 1:].plot.area(subplots=True, color='k', figsize=(25, 33))
plt.show()


##### Promedio por días ######
tabla['Promedio'] = tabla.mean(axis=1)
tabla['Promedio'].plot(title=nombre, color='k', figsize=(10, 6))
plt.xlabel('')
plt.xticks(etiquetax, fontsize=15)
plt.axvline(salida_sol, color='r')
plt.axvline(puesta_sol, color='r')

tabla.mean(axis=1).plot(title=nombre, color='k', figsize=(10, 6))
plt.xlabel('')
plt.xticks(etiquetax, fontsize=15)
plt.axvline(salida_sol, color='r')
plt.axvline(puesta_sol, color='r')

#######################################################################


##### Actograma #####
plt.figure(figsize=(15, 23), dpi=80)
plt.subplot(total_dias, 1, 1)
plt.plot(tabla.iloc[:, 0])

fig, ax = plt.subplots()


tabla.iloc[:, :-1].plot.bar(title=nombre, subplots=True)





datos1 = datos.pivot(index=datos.index, columns='Javier')



datos.resample('1D').ohlc()
########################################################################
datos1 = pd.date_range(intervalo, periods=len(datos), freq=frecu)
datos1 = datos1.dt.time.to_frame()
datos1 = pd.date_range(intervalo, periods=len(datos), freq=frecu).dt.time




datos.resample('1D').plot(kind='bar')

    

# Sacar sumatoria o promedio 
#Actograma
# graficar desde el segundo día

datos.resample('1D').mean().plot()


datos.resample('1D').values

type(nombre)
type(fecha)
type(hora)
type(intervalo)
type(frecu)
type(len_datos)
type(edad)
type(serie_acti)
type(sexo)

######################################################################
plt.figure(figsize=(8, 5), dpi=80)
plt.subplot(111)

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C = np.cos(X)
S = np.sin(X)

plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
plt.plot(X, S, color="red", linewidth=2.5, linestyle="-",  label="sine")

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

plt.xlim(X.min() * 1.1, X.max() * 1.1)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
          [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.ylim(C.min() * 1.1, C.max() * 1.1)
plt.yticks([-1, +1],
          [r'$-1$', r'$+1$'])

t = 2*np.pi/3
plt.plot([t, t], [0, np.cos(t)],
        color='blue', linewidth=1.5, linestyle="--")
plt.scatter([t, ], [np.cos(t), ], 50, color='blue')
plt.annotate(r'$sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
            xy=(t, np.sin(t)), xycoords='data',
            xytext=(+10, +30), textcoords='offset points', fontsize=16,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.plot([t, t], [0, np.sin(t)],
        color='red', linewidth=1.5, linestyle="--")
plt.scatter([t, ], [np.sin(t), ], 50, color='red')
plt.annotate(r'$cos(\frac{2\pi}{3})=-\frac{1}{2}$', xy=(t, np.cos(t)),
            xycoords='data', xytext=(-90, -50), textcoords='offset points',
            fontsize=16,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.legend(loc='upper left')

plt.show()



fecha_actual=datetime.datetime.now()
fecha_ayer=fecha_actual-datetime.timedelta(days=1)
fecha_manana=fecha_actual+datetime.timedelta(days=1)
# mostramos por pantalla las tres fechas
print("fecha actual: ", fecha_actual)
print("fecha ayer: ", fecha_ayer)
print("fecha mañana: ", fecha_manana)

#######################################################
#### FUNCIÓN RECTANGULAR ####
x = datos.iloc[:, 0]
plt.plot(x)
x[100:200] = 1
X = scipy.fftpack.fft(x)
f, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
ax0.plot(x)
ax0.set_ylim(-0.1, 1.1)
ax1.plot(scipy.fftpack.fftshift(np.abs(X)))
ax1.set_ylim(-5, 55)

######
a = np.zeros( (1000,) )   # whatever size. initializes to zeros
a[150:180] = 1.0
plt.plot(a)
plt.show()

N = 100 # sample count
P = 10  # period
D = 5   # width of pulse
sig = np.arange(N) % P < D
plt.plot(sig)

def rect(T):
    """create a centered rectangular pulse of width $T"""
    return lambda t: (-T/2 <= t) & (t < T/2)

def pulse_train(t, at, shape):
    """create a train of pulses over $t at times $at and shape $shape"""
    return np.sum(shape(t - at[:,np.newaxis]), axis=0)

sig = pulse_train(
    t=np.arange(100),              # time domain
    at=np.array([0, 10, 40, 80]),  # times of pulses
    shape=rect(10)                 # shape of pulse
)


from scipy import signal
import matplotlib.pyplot as plt
t = np.linspace(0, 1, 500, endpoint=False)
plt.plot(t, signal.square(2 * np.pi * 5 * t))
plt.ylim(-2, 2)


T=10
D=5
N=10
shift = 1/4   # number of cycles to shift (1/4 cycle in your example)
x = np.linspace(0, T*N, 10000, endpoint=False)
y=signal.square(2 * np.pi * (1/T) * x + 2*shift*np.pi)
plt.plot(x,y)
plt.ylim(-2, 2)
plt.xlim(0, T*N)

### Este es el mejor ejemplo de función rectangular
x = np.zeros(500)
x[100:200] = 1
X = scipy.fftpack.fft(x)
f, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
ax0.plot(x)
ax0.set_ylim(-0.1, 1.1)
ax1.plot(scipy.fftpack.fftshift(np.abs(X)))
ax1.set_ylim(-5, 55)
##############################################################