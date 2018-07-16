print('''
      PyGraph
      Desarrollado por Javier Villanueva-Valle,
      Estudiante de Doctorado en Neurociencias de la Conducta,
      Facultad de Psicología, UNAM.
      Ciudad de México, México.
      email: javier830409@gmail.com
      2018-Enero\n\n''')
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import datetime
from time import time
import os
import fnmatch
#####################################################
directorio = os.getcwd()
archivos_AWD = []
for AWD in os.listdir(directorio):
    if fnmatch.fnmatch(AWD, "*.AWD"):
        archivos_AWD.append(AWD)
print(archivos_AWD)
#####################################################
df = str(input('\nEscribe el nombre del archivo que aparece arriba SIN LAS COMILLAS. "ejemplo.AWD"\nArchivo -->  '))
tiempo_inicial = time()
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
elif total_dias in np.arange(11, 32, 1):
    tamaño = (10, 20)
    titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
elif total_dias in np.arange(31, 91, 1):
    tamaño = (10, 40)
    titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
elif total_dias in np.arange(91, 182, 1):
    tamaño = (10, 50)
    titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
elif total_dias in np.arange(182, 366, 1):
    tamaño = (12, 180)
    titulo = '\nActograma\n{}. {}. Edad {} años. {}.'.format(serie_acti, nombre, edad, sexo)
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