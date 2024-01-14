#Ejercicio 00
import pandas as pd
data = pd.read_csv('assets/real_estate.csv', sep=';')
#print(data.head())

#Ejercicio 01
precio_max = data["price"].max()
indice = data.loc[data['price'] == precio_max]
direcc = data.loc[13110]['address']
propiedad = "The house with address " + direcc + " is the most expensive and its price is " + str(precio_max) + " USD."
#print(propiedad)

#Ejercicio 02
#print(data.filter(['price']))
precio_min = data['price'].min(skipna=True)
indice = data.loc[data['price'] == precio_min]
#for i in indice:
    #print('The house with address ' + indice['address'] + ' is the cheapest and its price is ' + str(precio_min) + ' USD')
    #break
    

#Ejercicio 03
#print(data.filter(['surface']))
s_max = data['surface'].max(skipna = True)
s_min = data['surface'].min(skipna = True)
index_d_max = data['address'].loc[data['surface'] == s_max]
index_d_min = data['address'].loc[data['surface'] == s_min]
dmax = data.loc[6794]['address']
dmin = data.loc[498]['address']
#print('The bigger house is located on ' + dmax + ' and its surface is ' + str(s_max) + ' meters.')
#print('The smaller house is located on ' + dmin + ' and its surface is ' + str(s_min) + ' meters.')

#Ejercicio 04
columnas = data.columns.values
#print(columnas)
#print(data.filter(['level5']))
poblaciones = list(data.level5.unique())
#print(poblaciones)
#print(len(poblaciones))
zonas = list(set(data.level5))
#print(zonas)
#print(len(zonas))


#Ejercicio 05
por_columns = data.isnull().any()
#print(por_columns)
por_conjunto = data.isnull().any().any()
#print(por_conjunto)

#Ejercicio 06
#print("DataSet Original:")
#print(data)
data1 = data.dropna(axis=1)
#print("DataSet sin columnas que contienen al menos un valos nulo:")
#print(data1)
data2 = data.dropna()
#print("DataSet sin filas que contienen al menos un valor nulo:")
#print(data2)
data3 = data.dropna(axis=1, how = 'all')
#print("DataSet solo sin las columnas que tienen todos los valores nulos:")
#print(data3)
data4 = data.dropna(how = "all")
#print("DataSet solo sin las filas que tienen todos los valores nulos:")
#print(data4)

#Ejercicio 07

poblacion = ['Arroyomolinos (Madrid)']
data_arroyomolinos = data[data['level5'].isin(poblacion)]
precio_medio = data_arroyomolinos['price'].mean()
#print(precio_medio)

#Ejercicio 08

precio = data_arroyomolinos.filter(['price'])
import matplotlib.pyplot as plt
plt.figure(figsize = (10, 5))
plt.hist(precio, bins=50, alpha=0.8)
plt.title("Distribución de precios en Arroyomolinos (Madrid)")
plt.legend("Zona de poder adquisitivo medio-alto")
plt.show()

#Ejercicio 09
val = ['Valdemorillo']
gal = ['Galapagar']
val1 = data[data['level5'].isin(val)]
pm_valdemorillo = val1['price'].mean()
gal1 = data[data['level5'].isin(val)]
pm_galapagar = gal1['price'].mean()
if pm_valdemorillo < pm_galapagar:
    print("La zona de Galapagar es más valorada que la de Valdemorillo")
elif pm_valdemorillo > pm_galapagar:
    print("La zona de Valdemorillo es más valorada que la de Galapagar")
else:
    print("Las zonas de Valdemorillo y Galapagar tienen el mismo precio promedio")

#Ejercicio 10
m2_val = val1.loc[:, ['surface', 'price']]
m2_val['pps'] = m2_val['price'] / m2_val['surface']
promedio_m2val = m2_val['pps'].mean()
#print(promedio_m2val)

m2_gal = gal1.loc[:, ['surface', 'price']]
m2_gal['pps'] = m2_gal['price'] / m2_gal['surface']
promedio_m2gal = m2_gal['pps'].mean()
#print("El precio medio por metro cuadrado en Valdemorillo y Galapagar es el mismo: " + str(promedio_m2gal) + '€')

#Ejercicio 11
sp = list(data['surface'])
pr= list(data['price'])
plt.figure(figsize = (10, 5))
plt.scatter(sp, pr)
plt.title("Precio y superficie")
plt.show()

#Ejercicio 12
agencias = list(data.realEstate_name.unique())
#print('El numero de agencias en el dataset es: ' + str(len(agencias)))

#Ejercicio 13
rep = data['level5'].value_counts()
maxi = rep.max(skipna = True)
#print("La población con más inmuebles es Madrid Capital con " + str(maxi) + " en total.")

#Ejercicio 14
fuenla = ['Fuenlabrada']
leg = ['Leganés']
gtf = ['Getafe']
alco = ['Alcorcón']
new_data = data[data['level5'].isin(fuenla) + data['level5'].isin(leg) + data['level5'].isin(gtf) + data['level5'].isin(alco)]
#print(new_data['level5'])

#Ejercicio 15
mediafuenla = new_data[new_data['level5'].isin(fuenla)]
f = mediafuenla['price'].mean()

medialeg = new_data[new_data['level5'].isin(leg)]
l = medialeg['price'].mean()

mediagtf = new_data[new_data['level5'].isin(gtf)]
g = mediagtf['price'].mean()

medialco = new_data[new_data['level5'].isin(alco)]
a = medialco['price'].mean()

labels = ['Fuenlabrada', 'Leganés','Getafe', 'Alcorcón']
values = [f, l, g, a]
plt.figure(figsize = (10, 5))
plt.bar(labels, values)
plt.title('Gráficos precios promedio')
plt.legend('Getafe es la zona más cara, mientras que Fuenlabrada es la más barata')
plt.show()

#Ejercicio 16
z = list(new_data['price'])
precio_medio = new_data['price'].mean()
#print(precio_medio)
precio_varianza = new_data['price'].var(ddof = 0)
#print(precio_varianza)

    #Cuasi-varianza
#print(new_data['price'].var())

    #Varianza
#print(new_data['price'].var(ddof = 0))
    # == #
#from numpy import var
#z = list(new_data['price'])
#print(var(z))

z1 = list(new_data['rooms'])
r_medio = new_data['rooms'].mean()
r_var = new_data['rooms'].var(ddof = 0)
#print(r_medio)
#print(r_var)

z2 = list(new_data['surface'])
#print(z2)
s_medio = new_data['surface'].mean()
s_var = new_data['surface'].var(ddof = 0)
#print(s_medio)
#print(s_var)

b_medio = new_data['bathrooms'].mean()
b_var = new_data['bathrooms'].var(ddof = 0)
#print(b_medio)
#print(b_var)

k = (new_data.groupby(lambda _: "").agg(
   Precio = ('price', 'mean'),
   Habitaciones = ('rooms', 'mean'),
   Superficie = ('surface', 'mean'),
   Baños = ('bathrooms', 'mean'))
)

v = (new_data.groupby(lambda _: "").agg(
    Precio = ('price', 'var'),
    Habitaciones = ('rooms', 'var'),
    Superficie = ('surface', 'var'),
    Baños = ('bathrooms', 'var')
))

final = pd.concat([k, v], axis=0)
final.index = ['Media', 'Varianza']
#print(final)

#Ejercicio 17 - *REVISAR (Convertir en DataFrame)*

#fuenla = ['Fuenlabrada']
#leg = ['Leganés']
#gtf = ['Getafe']
#alco = ['Alcorcón']
Fuenlabrada = new_data[new_data['level5'].isin(fuenla)]
prixmax= Fuenlabrada['price'].max()
direcc = Fuenlabrada['address'].loc[Fuenlabrada['price'] == prixmax]
soluc = "The house with address " + direcc + " is the most expensive and its price is " + str(prixmax) + " USD."

Leganés = new_data[new_data['level5'].isin(leg)]
prmax= Leganés['price'].max()
direc = Leganés['address'].loc[Leganés['price'] == prmax]
solu = "The house with address " + direc + " is the most expensive and its price is " + str(prmax) + " USD."

Getafe = new_data[new_data['level5'].isin(gtf)]
pmax= Getafe['price'].max()
dir = Getafe['address'].loc[Getafe['price'] == pmax]
sol = "The house with address " + dir + " is the most expensive and its price is " + str(pmax) + " USD."

Alcorcón = new_data[new_data['level5'].isin(alco)]
pm= Alcorcón['price'].max()
dire = Alcorcón['address'].loc[Alcorcón['price'] == pm]
solc = "The house with address " + dire + " is the most expensive and its price is " + str(pm) + " USD."

#Ejercicio 18
  #Normalización media
Fuenlabrada = new_data[new_data['level5'].isin(fuenla)]
df = Fuenlabrada['price']
w = (df-df.mean())/df.std()

Leganés = new_data[new_data['level5'].isin(leg)]
dl = Leganés['price']
q = (dl-dl.mean())/dl.std()

Getafe = new_data[new_data['level5'].isin(gtf)]
dg= Getafe['price']
t = (dg-dg.mean())/dg.std()

Alcorcón = new_data[new_data['level5'].isin(alco)]
da = Alcorcón['price']
j = (da-da.mean())/da.std()

dathist = pd.concat([w, q, t, j], axis=1)
dathist.columns = ['Fuenlabrada', 'Leganés', 'Getafe', 'Alcorcón']

plt.figure(figsize = (10, 5))

plt.hist([w, q, t, j], bins=30, alpha=0.5, label=['w', 'q', 't', 'j'])

plt.title("Histograma normalización de precios")
plt.legend("Los valores de cada columna ahora están normalizados de modo que la media de los valores de cada columna sea 0 y la desviación estándar de los valores de cada columna sea 1")
plt.show()

#Ejercicio 19
gf = new_data[new_data['level5'].isin(gtf)]
ac = new_data[new_data['level5'].isin(alco)]
gf_2 = gf.loc[:, ['surface', 'price']]
gf_2['pps'] = gf_2['price'] / gf_2['surface']
ac_2 = ac.loc[:, ['surface', 'price']]
ac_2['pps'] = ac_2['price'] / ac_2['surface']
media_gf = gf_2['pps'].mean()
var_gf = gf_2['pps'].var(ddof=0)
media_ac = ac_2['pps'].mean()
var_ac = ac_2['pps'].var(ddof=0)
print('El precio medio por metro cuadrado es de ' + str(media_gf) + ' en Getafe, y de ' + str(media_ac) + ' en Alcorcón')
if media_gf < media_ac:
    print('Getafe es una zona más económica que Alcorcón')
elif media_gf > media_ac:
    print('Getafe es más cara que Alcorcón')
else:
    print('Getafe y Alcorcón tienen el mismo precio por m2')

if var_ac < var_gf:
    print('Los precios por m2 entre inmuebles son más dispares en Getafe')
elif var_ac > var_gf:
    print('Los precios por m2 entre inmuebles son más dispares en Alcorcón')
else:
    print('Hay la misma disparidad de precios entre inmuebles en ambas zonas')

#Ejercicio 20
    
#Fuenlabrada = new_data[new_data['level5'].isin(fuenla)]
xf = Fuenlabrada['price']
yf = Fuenlabrada['surface']
#Leganés = new_data[new_data['level5'].isin(leg)]
xl = Leganés['price']
yl = Leganés['surface']
#Getafe = new_data[new_data['level5'].isin(gtf)]
xg = Getafe['price']
yg = Getafe['surface']
#Alcorcón = new_data[new_data['level5'].isin(alco)]
xa = Alcorcón['price']
ya = Alcorcón['surface']

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (4, 2))

ax1.scatter(xf, yf)
ax1.set_xlabel('precio')
ax1.set_ylabel('superficie')

ax2.scatter(xl, yl)
ax2.set_xlabel('precio')
ax2.set_ylabel('superficie')

ax3.scatter(xg, yg)
ax3.set_xlabel('precio')
ax3.set_ylabel('superficie')

ax4.scatter(xa, ya)
ax4.set_xlabel('precio')
ax4.set_ylabel('superficie')

plt.show()

plt.figure(figsize = (10, 5))

plt.plot(xf, yf, label = 'Fuenlabrada')
plt.plot(xl, yl, label = 'Leganés')
plt.plot(xg, yg, label = 'Getafe')
plt.plot(xa, ya, label = 'Alcorcón')

plt.title('Gráfico')
plt.show()

coord = new_data.loc[:, ['latitude', 'longitude']]

latitud = coord['latitude'].to_numpy().transpose().tolist()
longitud = coord['longitude'].to_numpy().transpose().tolist()
diccionario = dict(zip(latitud, longitud))

#fig, ax = plt.subplots()

#for clave, valor in diccionario.items():
    
    #plt.scatter(clave, valor, edgecolors='red')

#plt.show()


#print(coordenadas)
#coord = coordenadas.to_dict('records')
#print(coord)

 


#fig, ax = plt.subplots()

#def mapping_data(coord):
   # x, y = [], []
    #for i in range(len(coord)):
        #x.append(coord[i][1])
        #y.append(coord[i][2])

    #return x, y

#y, x = mapping_data(coord)

#ax.scatter(x, y, edgecolors='red', linewidths=2, zorder=2)

new_data = data[data['level5'].isin(fuenla) + data['level5'].isin(leg) + data['level5'].isin(gtf) + data['level5'].isin(alco)]
coord_f = new_data[new_data['level5'].isin(fuenla)]
coordenadas = coord_f.loc[:, ['level5', 'latitude', 'longitude']]
#print(coordenadas.head())

#coord = new_data.loc[:, ['level5', 'latitude', 'longitude']]
#diccionario = coord.set_index('level5').to_dict(orient = 'index')
#print(diccionario)

coord_f = new_data[new_data['level5'].isin(fuenla)]
coordenadasF = coord_f.loc[:, ['latitude', 'longitude']]
latitudF = coordenadasF['latitude'].to_numpy().transpose().tolist()
longitudF = coordenadasF['longitude'].to_numpy().transpose().tolist()
diccionarioF = dict(zip(latitudF, longitudF))

coord_l = new_data[new_data['level5'].isin(leg)]
coordenadasL = coord_l.loc[:, ['latitude', 'longitude']]
latitudL = coordenadasL['latitude'].to_numpy().transpose().tolist()
longitudL = coordenadasL['longitude'].to_numpy().transpose().tolist()
diccionarioL = dict(zip(latitudL, longitudL))

coord_g = new_data[new_data['level5'].isin(gtf)]
coordenadasG = coord_g.loc[:, ['latitude', 'longitude']]
latitudG = coordenadasG['latitude'].to_numpy().transpose().tolist()
longitudG = coordenadasG['longitude'].to_numpy().transpose().tolist()
diccionarioG = dict(zip(latitudG, longitudG))

coord_a = new_data[new_data['level5'].isin(alco)]
coordenadasA = coord_a.loc[:, ['latitude', 'longitude']]
latitudA = coordenadasA['latitude'].to_numpy().transpose().tolist()
longitudA = coordenadasA['longitude'].to_numpy().transpose().tolist()
diccionarioA = dict(zip(latitudA, longitudA))

my_diccionario = {'Fuenlabrada' : diccionarioF, 'Leganés' : diccionarioG, 'Getafe' : diccionarioG, 'Alcorcón' : diccionarioA}
#print(my_diccionario)

for nombre, valor in my_diccionario.items():
    print("{}".format(nombre))
    for latitud, longitud in valor.items():
        print(" - {}: {}". format(latitud, longitud))

plt.figure(figsize = (10, 5))       
for nombre, valor in my_diccionario.items():
    for latitud, longitud in valor.items():
        ptl.scatter(latitud, longitud, label=nombre)


     

