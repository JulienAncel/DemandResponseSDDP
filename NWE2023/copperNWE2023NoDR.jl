using SDDP, GLPK, Plots, DelimitedFiles, CSV, DataFrames, Statistics, CPLEX


pathoutputs = "C:/Users/jujua/Desktop/M2_EEET/Stage CEC/NorthWestEurope/2023outputs/NoDRLP/"
pathdata = "C:/Users/jujua/Desktop/M2_EEET/Stage CEC/NorthWestEurope/Data/"

T = 364 # Nombre d'étapes
##########################################################################
# Donnees de demande
##########################################################################
pathdemres = pathdata*"scenarios_demande_residuelle.csv"
demande_resi = CSV.read(pathdemres, DataFrame)
scenarios = [[demande_resi[(t-1)*24+1:t*24,"Scen. $i"] for i in 1:20] for t in 1:T]
vec_proba = 1/20 * ones(20)

###########################################################################
# Paramètres exogènes --- Données à modifier et à prendre d'un fichier xlsx
###########################################################################
# Capping
PC = 3000 # €/MW

# Intercos
C_UK = 3780 #MW
C_Esp = 3000 #MW
C_Scand = 4609#MW
pathintercos=pathdata*"net_importsATSUPL.csv"
pattern_interco_SW_AT_PL = CSV.read(pathintercos,DataFrame)[:,"0"]

#apport deterministe=moyapports historiques imposés
V_UK = 95#
V_Esp = 96
V_Scand = 15

# Gaz COMPTE DEJA LES EFFICACITES
Vth_CCGT = 91.7  # €/MW
Cth_CCGT = 40109.8 # MW
Vth_TAG = 128.3 # €/MW 
Cth_TAG = 3081.88 # MW

# Charbon
Vth_lignite = 95.1
Cth_lignite = 6240
Vth_anthra = 165.1
Cth_anthra = 13731.6


# Nucleaire
Vnk = 23 # €/MW 
pathavnuke=pathdata*"nuke_availability.csv"
Ank = CSV.read(pathavnuke,DataFrame)[:,"0"] # % of nuke capa available at hour h in {1,..,8760}
Cnk = 67799 + 8447 + 1661# MW

# Hydrau
# Hydraulique conventionnel
E = 1 #efficiency
Vhy = 7.530 # €/MW
Chy = 8596 # MW
Shy = 182949020 #MWh

# Hydraulique pompé
Vhp = 9.540 # €/MW
Chp = 15244# MW
Shp = 45122412.6 #MWh

############################################################################
# Problème
############################################################################

graph = SDDP.LinearGraph(T)

function subproblem_builder(subproblem::Model, node::Int64)
    # State Variables
    @variable(subproblem, 0 <= Xhy[h=1:24] <= Shy , SDDP.State,initial_value= Shy)
    @variable(subproblem, 0 <= Xhp[h=1:24] <= Shp , SDDP.State,initial_value= Shp)
    
    #Control Variables
    @variable(subproblem, 0 <= imports_UK[h=1:24]  <= C_UK)
    @variable(subproblem, 0 <= exports_UK[h=1:24]  <= C_UK)
    @variable(subproblem, 0 <= imports_Esp[h=1:24]  <= C_Esp)
    @variable(subproblem, 0 <= exports_Esp[h=1:24]  <= C_Esp)
    @variable(subproblem, 0 <= imports_Scand[h=1:24]  <= C_Scand)
    @variable(subproblem, 0 <= exports_Scand[h=1:24]  <= C_Scand)

    @variable(subproblem, 0 <= gth_CCGT[h=1:24]  <= Cth_CCGT)
    @variable(subproblem, 0 <= gth_TAG[h=1:24]  <= Cth_TAG)

    @variable(subproblem, 0 <= gth_lignite[h=1:24]  <= Cth_lignite)
    @variable(subproblem, 0 <= gth_anthra[h=1:24]  <= Cth_anthra)

    @variable(subproblem, 0 <= gnk[h=1:24] <= Cnk * Ank[(node-1)*24+h])

    @variable(subproblem, 0 <= turbhy[h=1:24]  <= Chy*E)
    @variable(subproblem, 0 <= turbhp[h=1:24]  <= Chp*E)
    @variable(subproblem, 0 <= pumphp[h=1:24]  <= Chp*E)

    @variable(subproblem, 0 <= slackplus[h=1:24])
    @variable(subproblem, 0 <= slackmoins[h=1:24])
    
    # # Random variables 
    @variable(subproblem, demresi[h=1:24])
    Omega = scenarios[node]
    SDDP.parameterize(subproblem, Omega, vec_proba) do omicron
        for h in 1:24
            JuMP.fix(demresi[h],omicron[h])
        end
        return 
    end

    # Constraints
    @constraint(subproblem, demand_sat[h=1:24],
        0 == - demresi[h] + slackmoins[h] - slackplus[h] +
         gth_CCGT[h] + gth_TAG[h]+ gnk[h] + gth_lignite[h] + gth_anthra[h]+
         E * (turbhy[h] + turbhp[h] - pumphp[h]) +
         imports_UK[h] - exports_UK[h] + imports_Esp[h] - exports_Esp[h] + imports_Scand[h] - exports_Scand[h]+
         pattern_interco_SW_AT_PL[h] 
        )

    @constraint(subproblem, waterhy[h=1:24],
    Xhy[h].out == (h==1 ? Xhy[24].in : Xhy[h-1].out) - turbhy[h])
    @constraint(subproblem, waterhp[h=1:24],
    Xhp[h].out == (h==1 ? Xhp[24].in : Xhp[h-1].out) - turbhp[h] + pumphp[h])


    # Objective
    @stageobjective(subproblem, 
    sum(Vth_CCGT*gth_CCGT[h] + Vth_TAG*gth_TAG[h] + Vnk*gnk[h] + 
        Vth_lignite*gth_lignite[h] + Vth_anthra*gth_anthra[h]+
        E*(Vhy*turbhy[h] + Vhp*turbhp[h])+
        V_UK*(imports_UK[h] + exports_UK[h]) + 
        V_Esp*(imports_Esp[h] + exports_Esp[h]) +
        V_Scand*(imports_Scand[h] + exports_Scand[h])+
        PC*(slackplus[h]+slackmoins[h])  
        for h in 1:24)
    )
    return subproblem
end

model = SDDP.PolicyGraph(
    subproblem_builder,
    graph;
    sense = :Min,
    lower_bound=0,
    optimizer = CPLEX.Optimizer
)
model
##################################################################################################
# Entraînement
##################################################################################################

#SDDP.train(model, stopping_rules=[SDDP.Statistical(num_replications=100,iteration_period=5)])
#SDDP.train(model, stopping_rules=[SDDP.Shapiro(num_replications=200,iteration_period=5,epsilon=1e9)])
SDDP.train(model,iteration_limit=50)
##################################################################################################
# Sorties 
##################################################################################################
Nbsimus = 1000

# Variables à suivre

dictprix = Dict{Symbol,Function}(
    Symbol("price$h") => (sp::JuMP.Model) -> JuMP.dual(constraint_by_name(sp,"demand_sat[$h]")) for h in 1:24
)
# Simulations
println("Début simulations")
simulations = SDDP.simulate(model,
            Nbsimus,
            [:imports_UK,:exports_UK,:imports_Esp,:exports_Esp,:imports_Scand,:exports_Scand,:slackplus,:slackmoins,:Xhy,:Xhp,:turbhy,:turbhp,:pumphp,:gth_CCGT,:gth_TAG,:gnk,:gth_lignite,:gth_anthra],
            custom_recorders = dictprix)
println("Fin simulations")
# Export pour post-traitement
println("Ecriture prix...")
prix = zeros(Float64,T*24,Nbsimus)
for t in 1:T
    for h in 1:24
        for sim in 1:Nbsimus
            prix[(t-1)*24 + h,sim] = -simulations[sim][t][Symbol("price$h")]
        end
    end
end
writedlm("prix.txt",prix)
println("Ecriture prix : ok")

println("Ecriture coûts....")
cout = Array{Float64}(undef,T,Nbsimus)
for t in 1:T
    for sim in 1:Nbsimus
        cout[t,sim] = simulations[sim][t][:stage_objective]
    end
end
writedlm(pathoutputs*"cout.txt",cout)
println("Ecriture coûts : ok")

println("Ecriture decisions....")
Pumphp = Array{Float64}(undef,T*24 ,Nbsimus)
Turbhp = Array{Float64}(undef,T*24 ,Nbsimus)
Turbhy = Array{Float64}(undef,T*24 ,Nbsimus)
Gnk = Array{Float64}(undef,T*24 ,Nbsimus)
Gth_CCGT = Array{Float64}(undef,T*24 ,Nbsimus)
Gth_TAG = Array{Float64}(undef,T*24 ,Nbsimus)
Gth_lignite = Array{Float64}(undef,T*24 ,Nbsimus)
Gth_anthra = Array{Float64}(undef,T*24 ,Nbsimus)
ImpUK =  Array{Float64}(undef,T*24 ,Nbsimus)
ExpUK = Array{Float64}(undef,T*24 ,Nbsimus)
ImpEsp =  Array{Float64}(undef,T*24 ,Nbsimus)
ExpEsp = Array{Float64}(undef,T*24 ,Nbsimus)
ImpScand =  Array{Float64}(undef,T*24 ,Nbsimus)
ExpScand = Array{Float64}(undef,T*24 ,Nbsimus)
Slackp = Array{Float64}(undef,T*24,Nbsimus)
Slackm = Array{Float64}(undef,T*24,Nbsimus)
for t in 1:T
    for h in 1:24
        for sim in 1:Nbsimus
            Pumphp[(t-1)*24+h,sim] = simulations[sim][t][:pumphp][h]
            Turbhp[(t-1)*24+h,sim] = simulations[sim][t][:turbhp][h]
            Turbhy[(t-1)*24+h,sim] = simulations[sim][t][:turbhy][h]
            Gnk[(t-1)*24+h,sim] = simulations[sim][t][:gnk][h]
            Gth_CCGT[(t-1)*24+h,sim] = simulations[sim][t][:gth_CCGT][h]
            Gth_TAG[(t-1)*24+h,sim] = simulations[sim][t][:gth_TAG][h]
            Gth_lignite[(t-1)*24+h,sim] = simulations[sim][t][:gth_lignite][h]
            Gth_anthra[(t-1)*24+h,sim] = simulations[sim][t][:gth_anthra][h]
            ImpUK[(t-1)*24+h,sim] = simulations[sim][t][:imports_UK][h]
            ExpUK[(t-1)*24+h,sim] = simulations[sim][t][:exports_UK][h]
            ImpEsp[(t-1)*24+h,sim] = simulations[sim][t][:imports_Esp][h]
            ExpEsp[(t-1)*24+h,sim] = simulations[sim][t][:exports_Esp][h]
            ImpScand[(t-1)*24+h,sim] = simulations[sim][t][:imports_Scand][h]
            ExpScand[(t-1)*24+h,sim] = simulations[sim][t][:exports_Scand][h]
            Slackp[(t-1)*24+h,sim] = simulations[sim][t][:slackplus][h]
            Slackm[(t-1)*24+h,sim] = simulations[sim][t][:slackmoins][h]
        end
    end
end
writedlm(pathoutputs*"turbphp.txt",Turbhp)
writedlm(pathoutputs*"pumphp.txt",Pumphp)
writedlm(pathoutputs*"turbhy.txt",Turbhy)
writedlm(pathoutputs*"gnk.txt",Gnk)
writedlm(pathoutputs*"gth_CCGT.txt",Gth_CCGT)
writedlm(pathoutputs*"gth_TAG.txt",Gth_TAG)
writedlm(pathoutputs*"gth_lignite.txt",Gth_lignite)
writedlm(pathoutputs*"gth_anthra.txt",Gth_anthra)
writedlm(pathoutputs*"imports_UK.txt",ImpUK)
writedlm(pathoutputs*"exports_UK.txt",ExpUK)
writedlm(pathoutputs*"imports_Esp.txt",ImpEsp)
writedlm(pathoutputs*"exports_Esp.txt",ExpEsp)
writedlm(pathoutputs*"imports_Scand.txt",ImpScand)
writedlm(pathoutputs*"exports_Scand.txt",ExpScand)
writedlm(pathoutputs*"slackplus.txt",Slackp)
writedlm(pathoutputs*"slackmoins.txt",Slackm)
println("Ecriture decisions : ok")
