using SDDP, GLPK, Plots, DelimitedFiles, CSV, DataFrames, Statistics, CPLEX

pathoutputs = "C:/Users/jujua/Desktop/M2_EEET/Stage CEC/Sorties_remplissage/Sorties_M0/No_DR/"
pathdata = "C:/Users/jujua/Desktop/M2_EEET/Stage CEC/Data/"


T = 364 # Nombre d'étapes
##########################################################################
# Donnees de demande
##########################################################################
demande_resi = CSV.read(pathdata*"scenarios_demande_residuelle_2050_M0.csv", DataFrame)
scenarios = [[demande_resi[(t-1)*24+1:t*24,"Scen. $i"] for i in 1:20] for t in 1:T]
vec_proba = 1/20 * ones(20)

###########################################################################
# Paramètres exogènes --- Données à modifier et à prendre d'un fichier xlsx
###########################################################################
# Capping
PC = 3000 # €/MW

# Intercos
Cinterco = 45000 #MW
V_interco = 412 #€/MW
#prix à 201 €/MWh de sorte que l'appel à l'interco ne se fasse qu'en dernier recours

# Thermique
Vth_CCGT = 50  # €/MW conditions de gaz cher mais pas au niveau des pics de 2022
Cth_CCGT = 26845# MW
Vth_TAG = 200 # €/MW conditions de gaz cher mais pas au niveau des pics de 2022
Cth_TAG = 2655 # MW

# Nucleaire
Vnk = 23 # €/MW 
Ank = CSV.read(pathdata*"nuke_availability.csv",DataFrame)[:,"0"] # % of nuke capa available at hour h in {1,..,8760}
Cnk = 0 # MW

# Hydraulique
E = 0.9 #efficiency
remplissage_semaine = 1.35*CSV.read(pathdata*"MeanWaterFilling20152021.csv",DataFrame)[:,"0"]

# Hydraulique conventionnel
Vhy = 7.530 # €/MW
Chy = 22000 # MW
Shy_t = 28/37 * remplissage_semaine[1]
inflow_conv = 1.35*CSV.read(pathdata*"inflow_conv_2019.csv",DataFrame)[:,"0"]
maxhy = 28/37*maximum(remplissage_semaine)

# Hydraulique pompé
Vhp = 9.540 # €/MW
Chp = 8000 # MW
Shp_t = 9/37 * remplissage_semaine[1]
inflow_step = 1.35*CSV.read(pathdata*"inflow_step_2019.csv",DataFrame)[:,"0"]
maxhp= 9/37*maximum(remplissage_semaine)

############################################################################
# Problème
############################################################################

graph = SDDP.LinearGraph(T)

function subproblem_builder(subproblem::Model, node::Int64)
    # State Variables
    @variable(subproblem, 0 <= Xhy[h=1:24] , SDDP.State,initial_value= Shy_t[1])
    @variable(subproblem, 0 <= Xhp[h=1:24] <= Shp, SDDP.State,initial_value= Shp_t[1])
    
    #Control Variables
    @variable(subproblem, 0 <= imports[h=1:24]  <= Cinterco)
    @variable(subproblem, 0 <= exports[h=1:24]  <= Cinterco)

    @variable(subproblem, 0 <= gth_CCGT[h=1:24]  <= Cth_CCGT)
    @variable(subproblem, 0 <= gth_TAG[h=1:24]  <= Cth_TAG)

    @variable(subproblem, 0 <= gnk[h=1:24] <= Cnk * Ank[(node-1)*24+h])

    @variable(subproblem, 0 <= turbhy[h=1:24]  <= Chy*E)
    @variable(subproblem, 0 <= turbhp[h=1:24]  <= Chp*E)
    @variable(subproblem, 0 <= pumphp[h=1:24]  <= Chp*E)

    @variables(subproblem, begin
        0 <= slackplus[h=1:24]
        0 <= slackmoins[h=1:24]
    end)
    
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
    @constraint(
        subproblem, demand_sat[h=1:24],
        0 == - demresi[h] + slackmoins[h] - slackplus[h] + gth_CCGT[h] + gth_TAG[h]+ gnk[h] +
         E * (turbhy[h] + turbhp[h] - pumphp[h]) + imports[h] - exports[h]
        )

    @constraint(subproblem, waterhy[h=1:24],
    Xhy[h].out == (h==1 ? Xhy[24].in : Xhy[h-1].out) - turbhy[h]+ inflow_conv[(node-1)*24+h])

    @constraint(subproblem, waterhp[h=1:24],
    Xhp[h].out == (h==1 ? Xhp[24].in : Xhp[h-1].out) - turbhp[h] + pumphp[h]+ inflow_step[(node-1)*24+h])


    # Objective
    @stageobjective(subproblem, 
    sum(Vth_CCGT*gth_CCGT[h] + Vth_TAG*gth_TAG[h] + Vnk*gnk[h] + E*(Vhy*turbhy[h] + Vhp*turbhp[h])+ 
        V_interco*(imports[h]+exports[h])+ PC*(slackplus[h]+slackmoins[h])  
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

#SDDP.train(model, stopping_rules=[SDDP.Shapiro(num_replications=200,iteration_period=5,epsilon=1e9)])
SDDP.train(model,iteration_limit=75)
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
            [:imports,:exports,:slackplus,:slackmoins,:gth_CCGT,:gth_TAG,:gnk,:turbhy,:turbhp,:pumphp],
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
writedlm(pathoutputs*"prix.txt",prix)
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
Pumphp = Array{Float64}(undef,T*24,Nbsimus)
Turbhp = Array{Float64}(undef,T*24,Nbsimus)
Turbhy = Array{Float64}(undef,T*24,Nbsimus)
Gnk = Array{Float64}(undef,T*24,Nbsimus)
Gth_CCGT = Array{Float64}(undef,T*24,Nbsimus)
Gth_TAG = Array{Float64}(undef,T*24,Nbsimus)
Imp =  Array{Float64}(undef,T*24,Nbsimus)
Exp = Array{Float64}(undef,T*24,Nbsimus)
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
            Imp[(t-1)*24+h,sim] = simulations[sim][t][:imports][h]
            Exp[(t-1)*24+h,sim] = simulations[sim][t][:exports][h]
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
writedlm(pathoutputs*"imports.txt",Imp)
writedlm(pathoutputs*"exports.txt",Exp)
writedlm(pathoutputs*"slackplus.txt",Slackp)
writedlm(pathoutputs*"slackmoins.txt",Slackm)
println("Ecriture decisions : ok")