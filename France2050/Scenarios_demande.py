import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


annees = [2013 + i for i in range(8)]
def lecture(an):
    file = "Data/histoire_dem_res/{}_jour.csv".format(an)
    donnees = pd.read_csv(file)
    donnees = donnees[['Date','Heures','Jour','Consommation','Nucléaire','Eolien','Solaire','Hydraulique - Fil de l?eau + éclusée']]
    virer = []
    for i in range(len(donnees['Heures'])):
        if i%4!=0:
            virer.append(i)
    donnees.drop(virer,axis=0,inplace=True)
    return donnees
#####################################################################################
# DEFINE SCENARIO RTE 
#####################################################################################
#N03
target_eol = 65000 # EN MW
target_sol = 70000

#M0
#target_eol = 136000 # EN MW
#target_sol = 208000
####################################################################################
capa_eol = np.array([10300,10300,10322,11761,13569,12518,13610,16592]) #EN MW
capa_sol = np.array([5116,5628,6191,6772,7660,7170,8188,9438])
capa_fil = np.array([10300,10306,10314,10325.5,10326.98,11222,10955,9759])

target_fil = 0.57*22000 #part de 57% de fil de l eau dans hydraulique sans STEP
facteur_eol = 1+(target_eol-capa_eol)/capa_eol
facteur_sol = 1+(target_sol-capa_sol)/capa_sol
facteur_fil = 1+(target_fil-capa_fil)/capa_fil
print(facteur_eol)
capa_nuke_installee = 63130*np.ones(8) #MW
mult_futur = {'conso':1.35,
 'eolien': facteur_eol,
  'solaire': facteur_sol,
   'fil': facteur_fil}
dict_donnees = {}
dict_donnees_futures = {}
for i,an in enumerate(annees):
    mydata = lecture(an)
    total_res = mydata['Eolien'] + mydata['Solaire'] + mydata['Hydraulique - Fil de l?eau + éclusée']
    total_res_futur = mult_futur['eolien'][i] *  mydata['Eolien'] + mult_futur['solaire'][i] * mydata['Solaire'] + mult_futur['fil'][i] * mydata['Hydraulique - Fil de l?eau + éclusée']
    demfutur = mult_futur['conso'] * np.array( mydata['Consommation'])
    a_nuke = mydata['Nucléaire'] /capa_nuke_installee[an-2013]
    newdf = mydata[['Date','Heures','Jour','Consommation']]
    newdf['A_nuke'] = a_nuke
    newdf['RES'] = total_res
    dffutur = mydata[['Date','Heures','Jour']]
    dffutur['Consommation'] = demfutur
    dffutur['RES'] = total_res_futur
    dict_donnees[an] = newdf
    dict_donnees_futures[an] = dffutur



def index_20scenario():
    choix_demande = []
    for t in range(52):
        scenar_dem = []
        for y in range(2013,2021):
            scenar_dem.append([y,0])
            scenar_dem.append([y,1])
            scenar_dem.append([y,2])
        keepindex = np.random.choice(np.arange(24),size=20)
        final_scenar_dem=[]
        for i in range(20):
            final_scenar_dem.append(scenar_dem[keepindex[i]])
        choix_demande.append(final_scenar_dem)
    choix_res = []
    for t in range(52):
        scenar_res = []
        for y in range(2013,2021):
            scenar_res.append([y,0])
            scenar_res.append([y,1])
            scenar_res.append([y,-1])
        keepindex = np.random.choice(np.arange(24),size=20)
        final_scenar_res=[]
        for i in range(20):
            final_scenar_res.append(scenar_res[keepindex[i]])
        choix_res.append(final_scenar_res)       
    return choix_demande, choix_res



def build_scenario(actuel=False):
    choix_dem,choix_res = index_20scenario()
    cols = ["Scen. {}".format(i+1) for i in range(20)]
    df_demresi = pd.DataFrame(np.nan, index=[i for i in range(168*52)],columns=cols)
    df_res = pd.DataFrame(np.nan, index=[i for i in range(168*52)],columns=cols)
    for i in range(20):
        demande_resi= np.zeros(168*52)
        prod_res = np.zeros(168*52)
        for t in range(52): 
            source_year, add_day = choix_dem[t][i][0], choix_dem[t][i][1]
            source_year_res, add_day_res = choix_res[t][i][0], choix_res[t][i][1]
            if actuel:
                source_dem = list(dict_donnees[source_year]['Consommation'])
                source_res = list(dict_donnees[source_year_res]['RES'])
            else:
                source_dem = list(dict_donnees_futures[source_year]['Consommation'])
                source_res = list(dict_donnees_futures[source_year_res]['RES'])

            for h in range(168):
                date = 168*t+h+24*add_day
                if 0 <= date and date < len(source_dem)-1:
                    a = source_dem[date]
                else:
                    a = source_dem[168*t+h]
                date = 168*t+h+24*add_day_res
                if  0<= date and date< len(source_res)-1:
                    b = source_res[date]
                else:
                    b = source_res[168*t+h]
                demande_resi[168*t+h] = a-b
                prod_res[168*t+h] = b
        df_demresi["Scen. {}".format(i+1)] = demande_resi
        df_res["Scen. {}".format(i+1)] = prod_res
    df_res.to_csv("./Data/scenarios_res2020.csv")
    return df_demresi
            
av_nuke = np.zeros(168*52)
for i in range(168*52):
    av_nuke[i] = np.mean([list(dict_donnees[year]['A_nuke'])[i] for year in range(2013,2021)])
df_demresifutur = build_scenario()
df_avnuke = pd.DataFrame(av_nuke)

df_demresifutur.to_csv("./Data/scenarios_demande_residuelle_2050_N03.csv")
df_avnuke.to_csv("./Data/nuke_availability.csv")


plt.figure()
for i in range(10):
    plt.plot(np.arange(168),list(df_demresifutur["Scen. %s"%(2*i+1)]/1000)[:168])
plt.title("Residual demand scenarios - Winter week (2050-N03)")
plt.ylabel("Power demand (GW)")
plt.show()

plt.figure()
plt.plot(np.arange(168*52),av_nuke)
plt.title("Mean availability of nuclear plants")
plt.ylabel("Available share of installed capacity")
plt.show()