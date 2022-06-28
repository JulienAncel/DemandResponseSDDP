from nbformat import read
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import time
sns.set_theme(style="whitegrid",palette="muted")

Nbsimus=1000
PC_=3000
pathdr = "./2023outputs/With_DR/"
pathnodr = "./2023outputs/No_DR/"

pathsdr = ["./2023outputs/With_DR/","./2023outputs/DRLP/"]
scenarsdr=["DRHP","DRLP"]
paths = ["./2023outputs/No_DR/","./2023outputs/With_DR/","./2023outputs/NoDRLP/","./2023outputs/DRLP/"]
scenars=["NoDRHP","DRHP","NoDRLP","DRLP"]

def varWelfare(plot=True):
    coutnoDR = pd.read_csv(pathnodr+"cout.txt",sep="\t",header=None)
    coutdr = pd.read_csv(pathdr+"cout.txt",sep="\t",header=None)
    deltaW = (coutnoDR-coutdr).sum(axis="rows")/1e9 #en milliards d'euros
    deltaW.to_csv(pathdr+'deltaW.csv')
    if plot:
        plt.figure()
        plt.hist(np.array(deltaW),bins=125,density=True)
        plt.grid()
        plt.xlabel("Welfare delta (G€)")
        plt.savefig(pathdr+"deltaW_hist.png")
        plt.show()
    return deltaW.mean(), deltaW.std()

def price_at_winter_peak(dr=True):
    if dr:
        prix = pd.read_csv(pathdr+"prix.txt",sep="\t",header=None).clip(lower=0)
    else:
        prix = pd.read_csv(pathnodr+"prix.txt",sep="\t",header=None).clip(lower=0)
    heures=[]
    for h in range(8736):
        day=h // 24
        if day % 7 < 5 and h % 24 == 19:
            if day // 7 < 12 or day // 7 > 43:
                heures.append(h)
    prix19 = prix.iloc[heures].mean()
    if dr:
        prix19.to_csv(pathdr+'prix19.csv')
    else:
        prix19.to_csv(pathnodr+'prix19.csv')
    return prix19.mean(), prix19.std()

def plot_prices(numsim,periode=[0,8736],dr=True):
    if dr:
        prixdr = pd.read_csv(pathdr+"prix.txt",sep="\t",header=None).clip(lower=0)
        if numsim=='all':
            prixdr.iloc[periode[0]:periode[1]].plot()       
        else:
            prixdr[numsim].iloc[periode[0]:periode[1]].plot()
    else:
        prixnoDR = pd.read_csv(pathnodr+"prix.txt",sep="\t",header=None).clip(lower=0)
        if numsim=='all':
            prixnoDR.iloc[periode[0]:periode[1]].plot()
        else:
            prixnoDR[numsim].iloc[periode[0]:periode[1]].plot()
    plt.ylabel("Market Price (€/MWh)")
    plt.xlabel("Hours")
    plt.savefig(pathdr+"prix_h%s_to_h%s"%(periode[0],periode[1]))
    plt.show()
    return

def plot_mean_prices(periode=[0,8736]):
    prixdr = pd.read_csv(pathdr+"prix.txt",sep="\t",header=None).iloc[periode[0]:periode[1]].clip(lower=0)
    moyprix = np.array(prixdr.mean(axis=1))
    stdprix = np.array(prixdr.std(axis=1))
    moyplus = moyprix + 1.96*stdprix/np.sqrt(Nbsimus)
    moymoins = moyprix - 1.96*stdprix/np.sqrt(Nbsimus)
    prixnodr = pd.read_csv(pathnodr+"prix.txt",sep="\t",header=None).iloc[periode[0]:periode[1]].clip(lower=0)
    moyprixno = np.array(prixnodr.mean(axis=1))
    stdprixno = np.array(prixnodr.std(axis=1))
    moyplusno = moyprixno + 1.96*stdprixno/np.sqrt(Nbsimus)
    moymoinsno = moyprixno - 1.96*stdprixno/np.sqrt(Nbsimus)
    plt.figure()
    plt.plot(np.arange(periode[0],periode[1]),moyprixno,color='blue',label='No DR')
    plt.plot(np.arange(periode[0],periode[1]),moyplus,color='gray',alpha=0.5)
    plt.plot(np.arange(periode[0],periode[1]),moyprix,color='black',label='With DR')
    plt.plot(np.arange(periode[0],periode[1]),moymoins,color='gray',alpha=0.5)
    plt.ylabel("Mean Market Price (€/MWh)")
    plt.xlabel("Hours")
    plt.legend()
    plt.savefig(pathdr+"prix_moyen_h%s_to_h%s.png"%(periode[0],periode[1]))
    plt.show()
    return

def time_PC(path):
    prix = pd.read_csv(path+"prix.txt",sep="\t",header=None)
    hours_at_PC = 1*(prix >= PC_)
    meancumultimetraj = pd.DataFrame(hours_at_PC.cumsum().T.mean())
    meancumultimetraj.to_csv(path+'cumultimeatPC.csv')
    print("Mean cumulated time at PC %s"%hours_at_PC.sum().mean())
    time_at=pd.DataFrame(hours_at_PC.sum())
    time_at.to_csv(path+'timeatPC.csv')
    return hours_at_PC

def mix_moyen_rapide_interco(path,dr=True):
    if dr:
        print("Info DR")
        carac_shift=pd.read_csv("./Data/DR/carac_shifting.csv")
        carac_shed = pd.read_csv("./Data/DR/carac_shedding.csv")
        nshed = len(carac_shed["CapaInst"])
        nshift = len(carac_shift["CapaInst"])
        noms_shift = list(carac_shift['Unnamed: 0'])
        noms_shed = list(carac_shed['Unnamed: 0'])
        # generation
        print("Collecte generation")
        gthCCGT = pd.read_csv(path+"gth_CCGT.txt",sep="\t",header=None).T.mean()
        gthTAG = pd.read_csv(path+"gth_TAG.txt",sep="\t",header=None).T.mean()
        gthlig = pd.read_csv(path+"gth_lignite.txt",sep="\t",header=None).T.mean()
        gthAnt = pd.read_csv(path+"gth_anthra.txt",sep="\t",header=None).T.mean()
        gnk = pd.read_csv(path+"gnk.txt",sep="\t",header=None).T.mean()
        turbhp = pd.read_csv(path+"turbphp.txt",sep="\t",header=None).T.mean()
        turbhy = pd.read_csv(path+"turbhy.txt",sep="\t",header=None).T.mean()
        turbshed = [pd.read_csv(path+"turbshedTech%s.txt"%numtech,sep="\t",header=None).T.mean() for numtech in range(1,nshed+1)]
        turbshift = [pd.read_csv(path+"turbshiftTech%s.txt"%numtech,sep="\t",header=None).T.mean() for numtech in range(1,nshift+1)]
        imps = pd.read_csv(path+"imports_UK.txt",sep="\t",header=None).T.mean()+pd.read_csv(path+"imports_Esp.txt",sep="\t",header=None).T.mean()+pd.read_csv(path+"imports_Scand.txt",sep="\t",header=None).T.mean()
        # stockage
        print("Collecte stockage")
        pumphp = pd.read_csv(path+"pumphp.txt",sep="\t",header=None).T.mean()
        exps = pd.read_csv(path+"exports_UK.txt",sep="\t",header=None).T.mean()+pd.read_csv(path+"exports_Esp.txt",sep="\t",header=None).T.mean()+pd.read_csv(path+"exports_Scand.txt",sep="\t",header=None).T.mean()
        pumpshift = [pd.read_csv(path+"pumpshiftTech%s.txt"%numtech,sep="\t",header=None).T.mean() for numtech in range(1,nshift+1)]
    else:
        # generation
        print("Collecte generation")
        gthCCGT = pd.read_csv(path+"gth_CCGT.txt",sep="\t",header=None).T.mean()
        gthTAG = pd.read_csv(path+"gth_TAG.txt",sep="\t",header=None).T.mean()
        gthlig = pd.read_csv(path+"gth_lignite.txt",sep="\t",header=None).T.mean()
        gthAnt = pd.read_csv(path+"gth_anthra.txt",sep="\t",header=None).T.mean()
        gnk = pd.read_csv(path+"gnk.txt",sep="\t",header=None).T.mean()
        turbhp = pd.read_csv(path+"turbphp.txt",sep="\t",header=None).T.mean()
        turbhy = pd.read_csv(path+"turbhy.txt",sep="\t",header=None).T.mean()
        imps = pd.read_csv(path+"imports_UK.txt",sep="\t",header=None).T.mean()+pd.read_csv(path+"imports_Esp.txt",sep="\t",header=None).T.mean()+pd.read_csv(path+"imports_Scand.txt",sep="\t",header=None).T.mean()
        # stockage
        print("Collecte stockage")
        pumphp = pd.read_csv(path+"pumphp.txt",sep="\t",header=None).T.mean()
        exps = pd.read_csv(path+"exports_UK.txt",sep="\t",header=None).T.mean()+pd.read_csv(path+"exports_Esp.txt",sep="\t",header=None).T.mean()+pd.read_csv(path+"exports_Scand.txt",sep="\t",header=None).T.mean()
    my_gen = {'Nucléaire':gnk,
                'Thermique CCGT':gthCCGT,
                'Thermique TAG':gthTAG,
                'Lignite':gthlig,
                'Coal':gthAnt,
                'Hydrau. conv.':turbhy,
                'Hydrau. STEP': turbhp,
                'Imports':imps}
    my_stock = {'Hydrau. STEP':pumphp,'Exports':exps}
    if dr:
        for i in range(nshed):
            my_gen[noms_shed[i]+" - Shedding"] = turbshed[i]
        for i in range(nshift):
            my_gen[noms_shift[i]+" - Shifting"] = turbshift[i]
            my_stock[noms_shift[i]+" - Shifting"] = pumpshift[i]
    generation = pd.DataFrame(my_gen)
    stockage = pd.DataFrame(my_stock)
    return generation,stockage

def stack_plot_mix_periode_interco(mix,stock,demres,sigmadem,periode=[0,8736],allDRagg=True,quel='moyen',noDR=False):
    times = np.arange(periode[0],periode[1])
    nodr = "NODR" if noDR else ""
    carac_shift=pd.read_csv("./Data/DR/carac_shifting.csv")
    carac_shed = pd.read_csv("./Data/DR/carac_shedding.csv")
    nshed = len(carac_shed["CapaInst"])
    nshift = len(carac_shift["CapaInst"])
    noms_shift = list(carac_shift['Unnamed: 0'])
    noms_shed = list(carac_shed['Unnamed: 0'])
    if allDRagg:
        technos = ['Nucléaire', 'Thermique CCGT', 'Thermique TAG',"Lignite" , "Coal" , 'Hydrau. conv.', 'Hydrau. STEP', 'Imports', 'DR shift', 'DR shed']
        techstos = ['Hydrau. STEP', 'Exports', 'DR shift']
        ntechnos = ['Nuclear', 'Therm. CCGT', 'Therm. GT',"Lignite" , "Coal" , 'Conv. hydr.', 'PHS', 'Imports','DR shift', 'DR shed']
        ntechstos = ['PHS-demand','Exports','DR shift-demand']
        mix['DR shed'] = np.zeros(8736)
        for i in range(nshed):
            mix['DR shed'] += mix[noms_shed[i]+" - Shedding"]
        mix['DR shift'] = np.zeros(8736)
        stock['DR shift'] = np.zeros(8736)
        for i in range(nshift):
            mix['DR shift'] += mix[noms_shift[i]+" - Shifting"]
            stock['DR shift'] += stock[noms_shift[i]+" - Shifting"]

        Yfull = np.array([mix[tech].iloc[periode[0]:periode[1]] for tech in technos] )
        Yneg = np.array([-stock[tech].iloc[periode[0]:periode[1]] for tech in techstos])
    else:
        technos = ['Nucléaire', 'Thermique CCGT', 'Thermique TAG',"Lignite" , "Coal" , 'Hydrau. conv.', 'Hydrau. STEP','Imports']
        techstos = ['Hydrau. STEP','Exports']
        ntechnos = ['Nuclear', 'Therm. CCGT', 'Therm. GT',"Lignite" , "Coal" , 'Conv. hydr.', 'PHS','Imports']
        ntechstos = ['PHS-demand','Exports']
        Yfull = np.array([mix[tech].iloc[periode[0]:periode[1]] for tech in technos] )
        Yneg = np.array([-stock[tech].iloc[periode[0]:periode[1]] for tech in techstos])
    total = np.array(list(0.001*Yfull)+[0.001*stock[tech].iloc[periode[0]:periode[1]] for tech in techstos])
    colgene = ['blue','orange','red','brown','black','cyan','green','gray'] if noDR else ['blue','orange','red','brown','black','cyan','green','gray','purple','pink']
    colsto = ['limegreen','gray'] if noDR else ['limegreen','gray','orchid']
    plt.figure(figsize=(12,4))
    
    stacks_=plt.stackplot(times, 0.001*Yfull, labels=list(ntechnos),linewidth=0,colors=colgene)
    stacks=plt.stackplot(times, 0.001*Yneg, labels=list(ntechstos),linewidth=0,colors=colsto)
    hatches=['///' for i in range(len(colsto))]
    for stack, hatch in zip(stacks, hatches):
        stack.set_hatch(hatch)
        stack.set_edgecolor((0,0,0,0.5))
    stacks_[0].set_linewidth(2)
    stacks_[0].set_edgecolors('blue')
    stacks_[0].set_facecolors('white')
    if periode[1]-periode[0] != 8736:
        plt.plot(times,demres[periode[0]:periode[1]],label='Mean demand (+ det. net imports)',color='darkred',linestyle='-.',alpha=0.6)
        plt.plot(times,demres[periode[0]:periode[1]]+sigmadem[periode[0]:periode[1]],color='darkred',linestyle='-.',alpha=0.4)
        plt.plot(times,demres[periode[0]:periode[1]]-sigmadem[periode[0]:periode[1]],color='darkred',linestyle='-.',alpha=0.4)
    plt.xlabel("Hours")
    plt.ylabel("Generation in GW")
    plt.xlim(periode[0],periode[1]-2)
    plt.legend(loc='lower right',prop={'size':6})
    plt.tight_layout()
    plt.savefig(pathdr+"mixfull_"+nodr+quel+"_h%s_to_h%s.png"%(periode[0],periode[1]))
    plt.show()
    return

def emissions_mix_all_year(mix):
    emCO2=0
    # en MtCO2/MWh
    coef_em = {'Thermique CCGT':0.202/1e6, 'Thermique TAG':0.202/1e6, 'Lignite':0.428/1e6,'Coal':0.342/1e6}
    technos = ['Thermique CCGT', 'Thermique TAG',"Lignite","Coal"]
    for tech in technos:
        prod_tech = mix[tech].sum()
        emCO2 += prod_tech * coef_em[tech]
    return emCO2

def histo_emi(path):
    coef_em = {'Thermique CCGT':0.202/1e6, 'Thermique TAG':0.202/1e6, 'Lignite':0.428/1e6,'Anthra':0.342/1e6}
    gthCCGT = pd.read_csv(path+"gth_CCGT.txt",sep="\t",header=None).sum()
    gthTAG = pd.read_csv(path+"gth_TAG.txt",sep="\t",header=None).sum()
    gthlignite = pd.read_csv(path+"gth_lignite.txt",sep="\t",header=None).sum()
    gthanthra = pd.read_csv(path+"gth_anthra.txt",sep="\t",header=None).sum()
    emCO2=np.zeros(Nbsimus)
    for i in range(Nbsimus):
        emCO2[i]= gthTAG[i]*coef_em["Thermique TAG"] + gthCCGT[i]*coef_em["Thermique CCGT"] + gthlignite[i]*coef_em["Lignite"] + gthanthra[i]*coef_em["Anthra"]
    df = pd.DataFrame(emCO2)
    df.to_csv(path+'emissions.csv')
    # plt.figure()
    # plt.hist(emCO2,bins=40)
    # plt.title("Emissions - MtCO2eq")
    # plt.show()
    return df.mean(),df.std()

def analyse_profits(pathDR):
    carac_shift=pd.read_csv("./Data/DR/carac_shifting.csv")
    nshift = len(carac_shift["CapaInst"]) # nb of techs 
    carac_shed = pd.read_csv("./Data/DR/carac_shedding.csv")
    nshed = len(carac_shed["CapaInst"]) # nb of techs
    noms_shift = list(carac_shift['Unnamed: 0'])
    noms_shed = list(carac_shed['Unnamed: 0'])
    unit_shed = [[25,997] for i in range(nshed)]
    unit_shift = [[1246,1246],[1246,1246],[92851,92851],[189068,189068],[24927,112169],[24927,112169],[5840,7700],[5840,7700]]
    prix = pd.read_csv(pathDR+'prix.txt',sep='\t',header=None)
    print("Retrieve productions")
    productions_shed={}
    productions_shift={}
    for j,tech in enumerate(list(noms_shed)):
        print("Tech %s/%s"%(j+1,nshed))
        productions_shed[tech] = pd.read_csv(pathDR+"turbshedTech%s.txt"%(j+1),sep="\t",header=None)
    for j,tech in enumerate(list(noms_shift)):
        print("Tech %s/%s"%(j+1,nshift))
        productions_shift[tech] = [pd.read_csv(pathDR+"turbshiftTech%s.txt"%(j+1),sep="\t",header=None),pd.read_csv(pathDR+"pumpshiftTech%s.txt"%(j+1),sep="\t",header=None)]
    print("Calculate benefits")
    benefs={}
    total=np.zeros(Nbsimus)
    for j,tech in enumerate(list(noms_shed)):
        print("Tech %s/%s"%(j+1,nshed))
        activation = carac_shed['PrixAct'][j]
        tech_ben=np.zeros(Nbsimus)
        for i in tqdm(range(Nbsimus)):
            puissance_prod = np.array(productions_shed[tech][i])
            tech_ben[i] = np.sum((prix[i]-activation)*puissance_prod)/1e6
            total[i] += tech_ben[i]
        benefs[tech] = tech_ben
    for j,tech in enumerate(list(noms_shift)):
        print("Tech %s/%s"%(j+1,nshift))
        activation = carac_shift['PrixAct'][j]
        tech_ben=np.zeros(Nbsimus)
        for i in tqdm(range(Nbsimus)):
            puissance_prod=np.array(productions_shift[tech][0][i])
            puissance_rat=np.array(productions_shift[tech][1][i])
            tech_ben[i] = np.sum((prix[i]-activation)*puissance_prod - puissance_rat*prix[i] )/1e6
            total[i] += tech_ben[i]
        benefs[tech]=tech_ben
    print("Add total benefs")
    benefs['total'] = total
    df_ben = pd.DataFrame(benefs)
    df_ben.to_csv(pathDR+'benefices.csv')
    print("Substract installation costs")
    profitsb={}
    profitsh={}
    totb=total.copy()
    toth=total.copy()
    for j,tech in enumerate(list(noms_shed)):
        print("Tech %s/%s"%(j+1,nshed))
        coutb=unit_shed[j][0]*carac_shed["CapaInst"][j]/1e6
        profitsb[tech] = benefs[tech]- coutb
        totb-= coutb
        couth=unit_shed[j][1]*carac_shed["CapaInst"][j]/1e6
        profitsh[tech] = benefs[tech]- couth 
        toth-= couth
    for j,tech in enumerate(list(noms_shift)):
        print("Tech %s/%s"%(j+1,nshift))
        coutb=unit_shift[j][0]*carac_shift["CapaInst"][j]/1e6
        profitsb[tech] = benefs[tech]- coutb
        totb-=coutb 
        couth=unit_shift[j][1]*carac_shift["CapaInst"][j]/1e6
        profitsh[tech] = benefs[tech]- couth
        toth-=couth 
    profitsb['total']=totb
    profitsh['total']=toth
    gridsize=(3,4)
    fig=plt.figure(figsize=(15,15))
    for j,tech in enumerate(list(noms_shed)):
        r=j//4
        c=j%4
        ax = plt.subplot2grid(gridsize, (r,c))
        ax.boxplot([profitsb[tech],profitsh[tech]],showmeans=True,showfliers=False,positions=[1,1.5])
        ax.set_xticklabels(['',''],fontsize=5)
        ax.tick_params(axis='y',labelsize=7)
        ax.set_title(tech,fontsize=7 )
    for j,tech in enumerate(list(noms_shift)):
        r=(nshed+j)//4
        c=(nshed+j)%4
        ax = plt.subplot2grid(gridsize, (r,c))
        ax.boxplot([profitsb[tech],profitsh[tech]],showmeans=True,showfliers=False,positions=[1,1.5])
        ax.tick_params(axis='y',labelsize=7)
        if (nshed+j)//3==(nshed+nshift)//3:
            ax.set_xticklabels(['low','high'],fontsize=7)
        else:
            ax.set_xticklabels(['',''],fontsize=5)
        ax.set_title(tech,fontsize=7 )
    r = (nshed+nshift)//4
    c = (nshed+nshift)%4
    ax = plt.subplot2grid(gridsize, (r,c))
    ax.boxplot([profitsb['total'],profitsh['total']],showmeans=True,showfliers=False)
    ax.tick_params(axis='y',labelsize=7)
    ax.set_xticklabels(['low','high'],fontsize=7)
    ax.set_title('Total',fontsize=7 )
    fig.supylabel("Aggregated profits (M€)",fontsize=9)
    
    plt.savefig(pathDR+'bytech_profits.png')
    plt.show()
    
    df_profsb = pd.DataFrame(profitsb)
    df_profsb.to_csv(pathDR+'profits_low.csv')
    df_profsh = pd.DataFrame(profitsh)
    df_profsh.to_csv(pathDR+'profits_high.csv')
    return profitsb,profitsh

def profits_one_tech():
    techs = ['total','Steel','Aluminium','Chlorine','Cement','Paper and pulp',
    'Indus cooling','Cross-tech ventilation','Tertiary cooling','Tertiary heating',
    'Residential cooling','Residential heating']
    profits = {techs[i]:[] for i in range(len(techs))}
    by_scenar_h = {}
    by_scenar_l = {}
    for i,path in enumerate(pathsdr):
        profh = pd.read_csv(path+'profits_high.csv')
        profl = pd.read_csv(path+'profits_low.csv')
        by_scenar_h[scenarsdr[i]] = profh
        by_scenar_l[scenarsdr[i]] = profl
    for tech in techs:
        for scen in scenarsdr:
            profits[tech].append(by_scenar_h[scen][tech].to_list()+by_scenar_l[scen][tech].to_list())
    gridsize= (4,3)
    fig=plt.figure(figsize=(15,15))
    for j,tech in enumerate(techs[1:]):
        r=j//3
        c=j%3
        nticks = len(profits[tech])
        positions = np.arange(1,nticks+1)
        labels=scenarsdr
        to_plot = []
        for i in range(nticks):
            to_plot.append(profits[tech][i])
            
        ax = plt.subplot2grid(gridsize, (r,c))
        ax.boxplot(to_plot,showfliers=False,positions=positions)
        ax.set_xticklabels(labels,fontsize=5,rotation=0)
        ax.tick_params(axis='y',labelsize=4)
        title = tech 
        ax.set_title(title,fontsize=6,pad=-5,zorder=4)
    #total
    nticks = len(profits['total'])
    positions = np.arange(1,nticks+1)
    labels = []
    to_plot = []
    for i in range(nticks):
        to_plot.append(profits['total'][i])
        labels.append(scenarsdr[i])
    ax = plt.subplot2grid(gridsize, (r,c+1))
    ax.boxplot(to_plot,showfliers=False,positions=positions)
    ax.set_xticklabels(labels,fontsize=5,rotation=0)
    ax.tick_params(axis='y',labelsize=4)
    ax.set_title('Total',fontsize=6,pad=-5,zorder=4)
    fig.supylabel("Aggregated profits (M€)",fontsize=7)
    plt.subplots_adjust(top=0.9,
        bottom=0.1,
        left=0.075,
        right=0.9,
        hspace=0.4,
        wspace=0.135)
    plt.show()
    return profits

def benefs_one_tech():
    techs = ['total','Steel','Aluminium','Chlorine','Cement','Paper and pulp',
    'Indus cooling','Cross-tech ventilation','Tertiary cooling','Tertiary heating',
    'Residential cooling','Residential heating']
    profits = {techs[i]:[] for i in range(len(techs))}
    by_scenar_h = {}
    for i,path in enumerate(pathsdr):
        profh = pd.read_csv(path+'benefices.csv')
        by_scenar_h[scenarsdr[i]] = profh
    for tech in techs:
        for scen in scenarsdr:
            profits[tech].append(by_scenar_h[scen][tech].to_list())
    gridsize= (4,4)
    fig=plt.figure(figsize=(15,15))
    for j,tech in enumerate(techs[1:]):
        r=j//4
        c=j%4
        nticks = len(profits[tech])
        positions = np.arange(1,nticks+1)
        labels=scenarsdr
        to_plot = []
        for i in range(nticks):
            to_plot.append(profits[tech][i])
        ax = plt.subplot2grid(gridsize, (r,c))
        ax.boxplot(to_plot,showfliers=False,positions=positions)
        ax.set_xticklabels(labels,fontsize=5,rotation=0)
        ax.tick_params(axis='y',labelsize=4)
        title = tech 
        ax.set_title(title,fontsize=6,pad=-5,zorder=4)
    #total
    nticks = len(profits['total'])
    positions = np.arange(1,nticks+1)
    labels = []
    to_plot = []
    for i in range(nticks):
        to_plot.append(profits['total'][i])
        labels.append(scenarsdr[i])
    ax = plt.subplot2grid(gridsize, (r,c+1))
    ax.boxplot(to_plot,showfliers=False,positions=positions)
    ax.set_xticklabels(labels,fontsize=5,rotation=0)
    ax.tick_params(axis='y',labelsize=4)
    ax.set_title('Total',fontsize=6,pad=-5,zorder=4)
    fig.supylabel("Benefits (M€)",fontsize=7)
    plt.subplots_adjust(top=0.9,
        bottom=0.1,
        left=0.075,
        right=0.9,
        hspace=0.4,
        wspace=0.135)
    plt.show()
    return profits

def plot_levels(path,typedr,numtech,periode=[0,8736]):
    carac_shift=pd.read_csv("./Data/DR/carac_shifting.csv") 
    carac_shed = pd.read_csv("./Data/DR/carac_shedding.csv")
    noms_shift = list(carac_shift['Unnamed: 0'])
    noms_shed = list(carac_shed['Unnamed: 0'])
    nomtech = noms_shed[numtech-1] if typedr == 'shed' else noms_shift[numtech-1]
    correct = 1 if typedr=='shed' else 1
    data = pd.read_csv(path+"niveau"+typedr+"Tech%s.txt"%numtech,sep="\t",header=None) / correct
    m= data.mean(axis=1)
    s=data.std(axis=1)
    mup = m+s
    mdo = (m-s).clip(0)
    disclaimtype=" (contractual)"if typedr=='cont' else ""
    plt.figure()
    plt.plot(m.iloc[periode[0]:periode[1]],label="mean level")
    plt.plot(mup.iloc[periode[0]:periode[1]],color='gray',label="mean +/- s.d.")
    plt.plot(mdo.iloc[periode[0]:periode[1]],color='gray')
    plt.ylabel('Level in MWh')
    plt.title("Reservoir of technology: "+nomtech +disclaimtype)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(path+'meanlevel'+typedr+nomtech+'.png')
    plt.show()
    return

def usage_dr():
    carac_shift=pd.read_csv("./Data/DR/carac_shifting.csv")
    carac_shed = pd.read_csv("./Data/DR/carac_shedding.csv")
    noms_shift = list(carac_shift['Unnamed: 0'])
    noms_shed = list(carac_shed['Unnamed: 0'])
    usage={}
    for i,tech in enumerate(noms_shift):
        use=[]
        for path in pathsdr:
            use.append(pd.read_csv(path+'turbshiftTech%s.txt'%(i+1),sep="\t",header=None).T.mean().sum()/1e3)
        usage[tech] = use
    for i,tech in enumerate(noms_shed):
        use=[]
        for path in pathsdr:
            use.append(pd.read_csv(path+'turbshedTech%s.txt'%(i+1),sep="\t",header=None).T.mean().sum()/1e3)
        usage[tech] = use #GWh
    use=[]
    for path in pathsdr:
        use.append(pd.read_csv(path+'turbphp.txt',sep="\t",header=None).T.mean().sum()/1e3)
    usage['PHES'] = use
    cm=plt.get_cmap('tab20b')
    df_use=pd.DataFrame(usage,index=scenarsdr)
    df_use.plot.bar(stacked=True,linewidth=0,cmap=cm,rot=0)
    plt.legend()
    plt.ylabel("Mean Annual Turbined Energy (GWh)")
    plt.show()
    return df_use

def pumped_dr():
    carac_shift=pd.read_csv("./Data/DR/carac_shifting.csv")
    noms_shift = list(carac_shift['Unnamed: 0'])
    usage={}
    for i,tech in enumerate(noms_shift):
        use=[]
        for path in pathsdr:
            use.append(pd.read_csv(path+'pumpshiftTech%s.txt'%(i+1),sep="\t",header=None).T.mean().sum()/1e3)
        usage[tech] = use
    use=[]
    for path in pathsdr:
        use.append(pd.read_csv(path+'pumphp.txt',sep="\t",header=None).T.mean().sum()/1e3)
    usage['PHES'] = use
    cm=plt.get_cmap('tab20b')
    df_use=pd.DataFrame(usage,index=scenarsdr)
    df_use.plot.bar(stacked=True,linewidth=0,cmap=cm,rot=0)
    plt.legend()
    plt.ylabel("Mean Annual Pumped Energy (GWh)")
    plt.show()
    return df_use

def not_met_mean(path):
    slackplus = pd.read_csv(path+"slackplus.txt",sep="\t",header=None).T.mean()
    slackmoins = pd.read_csv(path+"slackmoins.txt",sep="\t",header=None).T.mean()
    p = slackplus.sum()/1e3
    m = slackmoins.sum()/1e3
    tot = p+m
    print("Mean unmet demand (GWh) : total %s -- positive %s -- negative %s"%(tot,p,m))
    return

def watval(path):
    dico_margval={}
    carac_shift=pd.read_csv("./Data/DR/carac_shifting.csv")
    carac_shed = pd.read_csv("./Data/DR/carac_shedding.csv")
    noms_shift = list(carac_shift['Unnamed: 0'])
    noms_shed = list(carac_shed['Unnamed: 0'])
    for i,tech in enumerate(noms_shed):
        dico_margval[tech]=pd.read_csv(path+"margvalshedTech%s.txt"%(i+1),sep="\t",header=None).T.mean()
    for i,tech in enumerate(noms_shift):
        dico_margval[tech]=pd.read_csv(path+"margvalshiftTech%s.txt"%(i+1),sep="\t",header=None).T.mean()
    dico_margval["Hydro conv."]=pd.read_csv(path+"margvalhy.txt",sep="\t",header=None).T.mean()
    dico_margval["Hydro STEP"]=pd.read_csv(path+"margvalhp.txt",sep="\t",header=None).T.mean()
    df = pd.DataFrame(dico_margval)
    prixdr = pd.read_csv(pathdr+"prix.txt",sep="\t",header=None).clip(lower=0)
    prixtout = prixdr.mean(axis=1)
    prix = np.array(prixtout.loc[prixtout.index%24==23])
    gridsize = (5,3)
    fig=plt.figure()
    c=0
    for tech in dico_margval:
        ax = plt.subplot2grid(gridsize, (c%5,c//5))
        c+=1
        ax.plot(dico_margval[tech])
        ax.plot(prix,color='black',alpha=0.1)
        ax.tick_params(axis='y',labelsize=4)
        ax.tick_params(axis='x',labelsize=4)
        ax.set_title(tech,fontsize=6,pad=-5,zorder=4)
    fig.supylabel("Water Value (€)",fontsize=7)
    fig.supxlabel("Day",fontsize=7)
    plt.savefig(path+"Dailywatervalues.png")
    plt.show()
    return df

def onewatval(path,typetech,numtech):
    val = pd.read_csv(path+"margvalshedTech%s.txt"%(numtech),sep="\t",header=None).T.mean() if typetech == 'shed' else pd.read_csv(path+"margvalshiftTech%s.txt"%(numtech),sep="\t",header=None).T.mean()
    prixdr = pd.read_csv(path+"prix.txt",sep="\t",header=None).clip(lower=0)
    prixtout = prixdr.mean(axis=1)
    prix = np.array(prixtout.loc[prixtout.index%24==23])
    plt.figure()
    plt.plot(val)
    plt.plot(prix,color='black',alpha=0.25)
    plt.ylabel("Water Value (€)")
    plt.xlabel("Day")
    plt.show()
    return

#onewatval("./2023outputs/With_DR/",'shift',1)

def onelevel(path,typetech,numtech):
    lev = pd.read_csv(path+"niveaushedTech%s.txt"%(numtech),sep="\t",header=None).T.mean()/1e3 if typetech == 'shed' else pd.read_csv(path+"niveaucontTech%s.txt"%(numtech),sep="\t",header=None).T.mean()/1e3
    lvl = np.array(lev.loc[lev.index%24==23])
    pac = pd.read_csv("./Data/DR/carac_shedding.csv")['PrixAct'].iloc[numtech-1] if typetech=='shed' else pd.read_csv("./Data/DR/carac_shifting.csv")['PrixAct'].iloc[numtech-1]
    val = pd.read_csv(path+"margvalshedTech%s.txt"%(numtech),sep="\t",header=None).T.mean() if typetech == 'shed' else pd.read_csv(path+"margvalshiftTech%s.txt"%(numtech),sep="\t",header=None).T.mean()
    prixdr = pd.read_csv(path+"prix.txt",sep="\t",header=None).clip(lower=0)
    prixtout = prixdr.mean(axis=1)
    prix = np.array(prixtout.loc[prixtout.index%24==23])
    diff= val+pac - prix
    ispos = diff>0
    indices = [] if ispos[0] else [0]
    for j in range(1,len(ispos)-1):
        if not ispos[j] and ispos[j-1]:
            indices.append(j)
        if ispos[j] and not ispos[j-1]:
            indices.append(j)
    plt.figure()
    plt.plot(lvl)
    for k in range(0,len(indices)-2,2):
            plt.axvspan(indices[k],indices[k+1],facecolor='0.5')
    plt.ylabel("Levels (GWh)")
    plt.xlabel("Day")
    plt.show()
    return
#onelevel("./2023outputs/With_DR/",'shift',1)


def levels(path):
    dico_lvl={}
    dico_diff={}
    m = pd.read_csv(path+"prix.txt",sep="\t",header=None).T.mean()
    marketprice = np.array(m.loc[m.index%24==23])
    carac_shift=pd.read_csv("./Data/DR/carac_shifting.csv")
    carac_shed = pd.read_csv("./Data/DR/carac_shedding.csv")
    noms_shift = list(carac_shift['Unnamed: 0'])
    noms_shed = list(carac_shed['Unnamed: 0'])
    for i,tech in enumerate(noms_shed):
        a=pd.read_csv(path+"niveaushedTech%s.txt"%(i+1),sep="\t",header=None).T.mean()/1e3
        dico_lvl[tech]=a.loc[a.index%24==23]
        b=np.array(pd.read_csv(path+"margvalshedTech%s.txt"%(i+1),sep="\t",header=None).T.mean())
        diff= b+carac_shed['PrixAct'].iloc[i] - marketprice
        ispos = diff>0
        indices = [] if ispos[0] else [0]
        for j in range(1,len(ispos)-1):
            if not ispos[j] and ispos[j-1]:
                indices.append(j)
            if ispos[j] and not ispos[j-1]:
                indices.append(j)
        dico_diff[tech]=indices
    for i,tech in enumerate(noms_shift):
        a=pd.read_csv(path+"niveaucontTech%s.txt"%(i+1),sep="\t",header=None).T.mean()/1e3
        dico_lvl[tech]=a.loc[a.index%24==23]
        b=np.array(pd.read_csv(path+"margvalshiftTech%s.txt"%(i+1),sep="\t",header=None).T.mean())
        diff= b+carac_shift['PrixAct'].iloc[i] - marketprice
        ispos = diff>0
        indices = [] if ispos[0] else [0]
        for j in range(1,len(ispos)-1):
            if not ispos[j] and ispos[j-1]:
                indices.append(j)
            if ispos[j] and not ispos[j-1]:
                indices.append(j)
        dico_diff[tech]=indices
    df = pd.DataFrame(dico_lvl)
    gridsize = (5,3)
    fig=plt.figure()
    c=0
    for tech in dico_lvl:
        ax = plt.subplot2grid(gridsize, (c%5,c//5))
        c+=1
        ax.plot(np.arange(364),dico_lvl[tech])
        ind = dico_diff[tech]
        for k in range(0,len(ind)-2,2):
            ax.axvspan(ind[k],ind[k+1],facecolor='0.5')
        ax.tick_params(axis='y',labelsize=4)
        ax.tick_params(axis='x',labelsize=4)
        ax.set_title(tech,fontsize=6,pad=-5,zorder=4)
    fig.supylabel("Levels (GWh)",fontsize=7)
    fig.supxlabel("Day",fontsize=7)
    plt.savefig(path+"Dailylevels.png")
    plt.show()
    return df

def origine_imports_exports(path,periode):
    dic_imports = {'UK':pd.read_csv(path+'imports_UK.txt',sep="\t",header=None).T.mean()/1e3,
    'Spain':pd.read_csv(path+'imports_Esp.txt',sep="\t",header=None).T.mean()/1e3,
    'Scandinavia':pd.read_csv(path+'imports_Scand.txt',sep="\t",header=None).T.mean()/1e3,
    'AT+PL+CH':pd.read_csv('./Data/importsATSUPL.csv')['0']/1e3}
    dic_exports = {'UK':-pd.read_csv(path+'exports_UK.txt',sep="\t",header=None).T.mean()/1e3,
    'Spain':-pd.read_csv(path+'exports_Esp.txt',sep="\t",header=None).T.mean()/1e3,
    'Scandinavia':-pd.read_csv(path+'exports_Scand.txt',sep="\t",header=None).T.mean()/1e3,
    'AT+PL+CH':-pd.read_csv('./Data/exportsATSUPL.csv')['0']/1e3}
    plotpos = np.array([dic_imports[key].iloc[periode[0]:periode[1]] for key in dic_imports])
    plotneg = np.array([dic_exports[key].iloc[periode[0]:periode[1]] for key in dic_exports])
    cols=['blue','red','cyan','green']
    plt.figure(figsize=(12,4))
    plt.stackplot(np.arange(periode[0],periode[1]), plotpos, labels=list(dic_exports.keys()),linewidth=0,colors=cols)
    plt.stackplot(np.arange(periode[0],periode[1]), plotneg,linewidth=0,colors=cols)
    plt.xlabel("Hours")
    plt.ylabel("Net Imports (GW)")
    plt.xlim(periode[0],periode[1]-2)
    plt.legend(loc='lower right',prop={'size':6})
    plt.tight_layout()
    plt.savefig(path+"net_imports_h%s_to_h%s.png"%(periode[0],periode[1]))
    #plt.show()
    uk=(dic_imports["UK"]+dic_exports["UK"]).sum()
    sp=(dic_imports["Spain"]+dic_exports["Spain"]).sum()
    sc=(dic_imports["Scandinavia"]+dic_exports["Scandinavia"]).sum()
    print(uk)
    print("Net yearly total imports (GWh): \n-- Uk %s\n--  Spain %s\n-- Sand %s"%(uk,sp,sc))
    return

def run(dr=True):
    if dr:
        mix = mix_moyen_rapide_interco(pathdr,dr=True)
        hydroconv = mix[0]['Hydrau. conv.'].sum()
        print("Hydro conv %s"%hydroconv)
        stack_plot_mix_periode_interco(mix[0],mix[1],demres,sigmademres,[4872,5039])
        stack_plot_mix_periode_interco(mix[0],mix[1],demres,sigmademres,[168,336])
        stack_plot_mix_periode_interco(mix[0],mix[1],demres,sigmademres,[8399,8567])
        stack_plot_mix_periode_interco(mix[0],mix[1],demres,sigmademres)
        m,s=histo_emi(pathdr)
        print("Emissions in average mix per year : %s MtCO2eq (%s)"%(m.to_list()[0],s.to_list()[0]))
        m,s=varWelfare()
        print("Mean and spread of var Welfare : %s -- %s"%(m,s))
        m,s=price_at_winter_peak(dr=True)
        print("Mean and spread of 7pm prices with DR : %s -- %s"%(m,s))
        time_PC(pathdr)
        not_met_mean(pathnodr)
        watval(pathdr)
        profb,profh = analyse_profits(pathdr)
        plot_mean_prices()
        plot_mean_prices([168,336])
        plot_mean_prices([4872,5039])
        plot_mean_prices([50*168,51*168])
        levels(pathdr)
        origine_imports_exports(pathdr,periode=[168,336])
        origine_imports_exports(pathdr,periode=[0,8736])
    else:
        mix = mix_moyen_rapide_interco(pathnodr,dr=False)
        hydroconv = mix[0]['Hydrau. conv.'].sum()
        print("Hydro conv %s"%hydroconv)
        stack_plot_mix_periode_interco(mix[0],mix[1],demres,sigmademres,[4872,5039],allDRagg=False,noDR=True)
        stack_plot_mix_periode_interco(mix[0],mix[1],demres,sigmademres,[168,336],allDRagg=False,noDR=True)
        stack_plot_mix_periode_interco(mix[0],mix[1],demres,sigmademres,[8399,8567],allDRagg=False,noDR=True)
        stack_plot_mix_periode_interco(mix[0],mix[1],demres,sigmademres,allDRagg=False,noDR=True)
        m,s=histo_emi(pathnodr)
        print("Emissions in average mix per year : %s MtCO2eq (%s)"%(m.to_list()[0],s.to_list()[0]))
        m,s=price_at_winter_peak(dr=False)
        print("Mean and spread of 7pm prices : %s -- %s"%(m,s))
        time_PC(pathnodr)
        not_met_mean(pathnodr)
        origine_imports_exports(pathnodr,periode=[168,336])
        origine_imports_exports(pathnodr,periode=[0,8736])
    return

print("############ NWE No DR ################")
dem_res = pd.read_csv("./Data/scenarios_demande_residuelle.csv").drop('Unnamed: 0',axis='columns')/1e3 #GW
net_imps = pd.read_csv("./Data/net_importsATSUPL.csv").drop('Unnamed: 0',axis='columns')/1e3
demres=np.array(dem_res.mean(axis=1)) + np.array(net_imps['0'].to_list())
sigmademres = np.array(dem_res.std(axis=1))
run(dr=False)

pathnodr = "./2023outputs/No_DR/"
print("############ NWE With DR ################")
run()


print("############ NWE 01 DR ################")
pathdr = "./2023outputs/01_DR/"
run()

print("############ NWE 05 DR ################")
pathdr = "./2023outputs/05_DR/"
run()

print("############ NWE 075 DR ################")
pathdr = "./2023outputs/075_DR/"
run()

print("############ NWE NoDRLP ################")
pathnodr = "./2023outputs/NoDRLP/"
run(dr=False)

print("############ NWE DRLP ################")
pathdr = "./2023outputs/DRLP/"
run()

print("############### Bilan ############################")
pumped_dr()
usage_dr()
profits_one_tech()
benefs_one_tech()


emissions=[]
cumtimes=[]
prix19s=[]
for i,path in enumerate(paths):
    ems = pd.read_csv(path+'emissions.csv')['0'].to_list()
    emissions.append(ems)
    cumtimePC = pd.read_csv(path+'timeatPC.csv')['0'].to_list()
    cumtimes.append(cumtimePC)
    prix19 = pd.read_csv(path+'prix19.csv')['0'].to_list()
    prix19s.append(prix19)

deltaws=[]
for i,p in enumerate(pathsdr):
    dw=pd.read_csv(p+'deltaW.csv')['0']
    deltaws.append(dw)

fig,ax=plt.subplots(2,2,figsize=(6,5))
ax[0,0].boxplot(deltaws,showmeans=True)
ax[0,0].tick_params(axis='y',labelsize=7)
ax[0,0].set_xticklabels(scenarsdr,fontsize=5)
ax[0,0].set_ylabel('Welfare variations (G€)',fontsize=7)

ax[0,1].boxplot(cumtimes,showmeans=True)
ax[0,1].tick_params(axis='y',labelsize=7)
ax[0,1].set_xticklabels(scenars,fontsize=5)
ax[0,1].set_ylabel('Total time at PC (h)',fontsize=7)

ax[1,0].boxplot(prix19s,showmeans=True)
ax[1,0].tick_params(axis='y',labelsize=7)
ax[1,0].set_xticklabels(scenars,fontsize=5)
ax[1,0].set_ylabel('Winter 7pm prices (€/MWh)',fontsize=7)

ax[1,1].boxplot(emissions,showmeans=True)
ax[1,1].tick_params(axis='y',labelsize=7)
ax[1,1].set_xticklabels(scenars,fontsize=5)
ax[1,1].set_ylabel('Emissions (MtCO2eq)',fontsize=7)
plt.tight_layout()
plt.show()