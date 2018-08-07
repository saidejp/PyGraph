[![Coverage Status](https://coveralls.io/repos/github/Sivlemx/PyGraph/badge.svg?branch=master)](https://coveralls.io/github/Sivlemx/PyGraph?branch=master)

# PyGraph

Software para analizar periodicidad y ritmicidad ciardiana en humanos y animales. 

Desarrollado por Javier Villanueva-Valle
email: javier830409@gmail.com

Reseña escrita el lunes, 16. julio 2018 04:02 

## En Español

### Objetivo

Actualizar Pgraph.

#### Pgrahp
Programa diseñado en 1992 para analizar la periodicidad y ritmos cicardianos. Basado en el artículo de **Dörrscheidt, G. J. & Beck, L., (1975), *Advanced methods for evaluating characteristic parameters ταρ of circadian rhythms*, Journal of Mathematical Biology 2(2):107-121** ([Circadian Rhythms](https://www.researchgate.net/publication/226211468_Advanced_methods_for_evaluating_characteristic_parameters_tar_of_circadian_rhythms)). 

Se desarrolló a principios de la decada de 1990, escrito en el lenguaje de programación **Fortran y C**. 

#### Descripción
* Corre bajo sistemas operativos Windows 95, 98, Me y XP.
* 50M en RAM
* 2.1M de espacio.
* Monitor, teclado y mouse.

Pgraph analiza datos obtenidos por los dispositivos Actiwatch [Actiware](http://www.actigraphy.com/solutions/actiware/) con extensión "AWD". Pero primero tiene que pasar por varios procedimientos:

#### Procedimiento
1. Se descargan los datos de los dispositivos Actiwatch en el sistema Win95, 98, Me y XP.
2. Los archivos con extensión "*.AWD*" tienen que convertirse en archivos con extensión binaria con el programa "Unformat".
(Unformat estandariza los datos de los archivos "*.AWD*" a 999. Es decir, mendiante la regla de tres el puntaje máximo dentro del archivo es a 999 como los demás es lo proporcional al dato máximo.)
3. El archivo transformado corre dentro de Pgraph.

#### Limitaciones
Únicamente corre en sistemas operativos Windows 95, 98, Me y XP. Debido al avance tecnológico Pgraph no es capaz de correr en Win10.

#### Necesidades
Pgrahp es necesario para aquellos que hacen o investigan ritmicidad cicardiana y/o periodicidad en humanos y animales. Por lo tanto, es necesario la actualización a sistemas operativos actuales (Linux, Mac, Win10).

### PyGraph
Nació bajo para cubrir esta necesidad en el año 2017 como parte de la formación del doctorado en neurociencias de la conducta de la facultad de psicología de la UNAM.
PyGraph está escrito desde Python 3.6 dentro de "Spyder".
Se utilizó Python por su sintaxis simple, clara y sencilla. Además de su facilidad de uso, legibilidad del código, facilidad de uso en dispositivos y la abundancia de bibliotecas.

#### Ventajas
Esto quiere decir que con el lenguaje de programación Python se eliminarán el procedimiento y las limitaciones que tiene PGraph. Dando paso a la constante implementación de análisis de de periodicidad, análisis de series de tiempo, estructuración de base de datos para exportar, generación de actogramas de más de 365 días consecutivos y muchas más aplicaciones ejecutados en varios scripts con el mínimo consumo de recursos de la computadora.

--------------------------------------------------

## English

### Objective

Update Pgraph.

#### Pgrahp
Software designed in 1992 to analyze the periodicity and rhythms of the Cicardians. Based on the article by **Dörrscheidt, GJ & Beck, L., (1975), *Advanced methods for evaluating characteristic parameters ταρ of circadian rhythms*, Journal of Mathematical Biology 2 (2): 107-121** [Circadian Rhythms](https://www.researchgate.net/publication/226211468_Advanced_methods_for_evaluating_characteristic_parameters_tar_of_circadian_rhythms) 

It was developed at the beginning of the 1990s, written in the programming language ** Fortran and C **.

#### Description
* Run under Windows 95, 98, Me and XP operating systems.
* 50M in RAM
* 2.1M of space.
* Monitor, keyboard and mouse.

Pgraph analyzes data obtained by Actiwatch devices [Actiware](http://www.actigraphy.com/solutions/actiware/) with extension "AWD". But first you have to go through several procedures:

#### Process
1. The data of the Actiwatch devices are downloaded in the Win95, 98, Me and XP system.
2. Files with extension "* .AWD *" must be converted into files with a binary extension with the "Unformat" program.
(Unformat standardizes the data from the files "* .AWD *" to 999. That is, by means of the rule of three the maximum score within the file is 999 as the rest is proportional to the maximum data.)
3. The transformed file runs inside Pgraph.

#### Limitations
It only runs on Windows 95, 98, Me and XP operating systems. Due to technological advance Pgraph is not able to run on Win10.

#### Needs
Pgrahp is necessary for those who do or investigate cycdic rhythmicity and / or periodicity in humans and animals. Therefore, it is necessary to upgrade to current operating systems (Linux, Mac, Win10).

### PyGraph
Created to cover this need in 2017 as part of the formation of the doctorate in behavioral neurosciences of the faculty of psychology of the UNAM.
PyGraph is written from Python 3.6 inside "Spyder".
Python was used for its simple, clear and simple syntax. In addition to its ease of use, readability of the code, ease of use in devices and the abundance of libraries.

#### Advantage
This means that the Python programming language will eliminate the procedure and limitations that PGraph has. Giving way to the constant implementation of analysis of periodicity, analysis of time series, structuring of database to export, generation of actograms of more than 365 consecutive days and many more applications executed in several scripts with the minimum consumption of resources of Computer.