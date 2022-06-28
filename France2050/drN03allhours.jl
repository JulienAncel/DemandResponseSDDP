using SDDP, GLPK, Plots, DelimitedFiles, CSV, DataFrames, Statistics,CPLEX

pathoutputs = "C:/Users/jujua/Desktop/M2_EEET/Stage CEC/Sorties_remplissage/Sorties_N03/With_DR/"
pathdata = "C:/Users/jujua/Desktop/M2_EEET/Stage CEC/Data/"

T = 364 # Nombre d'étapes
##########################################################################
# Donnees de demande
##########################################################################
demande_resi = CSV.read(pathdata*"scenarios_demande_residuelle_2050_N03.csv", DataFrame)
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

# Thermique
Vth_CCGT = 50  # €/MW 
Cth_CCGT = 455# MW
Vth_TAG = 200 # €/MW 
Cth_TAG = 45 # MW

# Nucleaire
Vnk = 23 # €/MW 
Ank = CSV.read(pathdata*"nuke_availability.csv",DataFrame)[:,"0"] # % of nuke capa available at hour h in {1,..,8760}
Cnk = 51000 # MW

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


# DR - Shedding
carac_shed = CSV.read(pathdata*"loadprofiles2050/carac_shedding.csv",DataFrame)
nshed = length(carac_shed[:,"CapaInst"]) # nb of techs
Cdr_shed = carac_shed[:,"CapaInst"] # MW
Vdr_shed = carac_shed[:,"PrixAct"] # €/MW
Ddr_shed = carac_shed[:,"Duree"]  # h
N_shed = carac_shed[:,"NAct"] # SU
S_shed = N_shed .* Ddr_shed .* Cdr_shed # MWh
A_shed = CSV.read(pathdata*"loadprofiles2050/loadshedding_availability.csv",DataFrame)

# DR - Shifting
carac_shift = CSV.read(pathdata*"loadprofiles2050/carac_shifting.csv",DataFrame)
nshift = length(carac_shift[:,"CapaInst"]) # nb of techs
Cdr_shift = carac_shift[:,"CapaInst"]
Vdr_shift = carac_shift[:,"PrixAct"]
Ddr_shift = carac_shift[:,"Duree"]
N_shift = carac_shift[:,"NAct"]
S_shift = Ddr_shift .* Cdr_shift
S_cont = S_shift .* N_shift
A_shift_turb = CSV.read(pathdata*"loadprofiles2050/loadshifting_turb_availability.csv",DataFrame)
A_shift_pump = CSV.read(pathdata*"loadprofiles2050/loadshifting_pump_availability.csv",DataFrame)

############################################################################
# Problème
############################################################################

graph = SDDP.LinearGraph(T)

function subproblem_builder(subproblem::Model, node::Int64)
    # State Variables
    @variable(subproblem, 0 <= Xshiftup[i=1:nshift,h=1:24] <= S_shift[i], SDDP.State,initial_value= S_shift[i] )
    @variable(subproblem, 0 <= Xshiftdo[i=1:nshift,h=1:24] <= S_shift[i], SDDP.State,initial_value= 0)
    @variable(subproblem, 0 <= Xshiftcont[i=1:nshift,h=1:24], SDDP.State,initial_value=S_cont[i] )

    @variable(subproblem, 0 <= Xshed[i=1:nshed,h=1:24] , SDDP.State,initial_value= S_shed[i] )
    
    @variable(subproblem, 0 <= Xhy[h=1:24] <= maxhy , SDDP.State,initial_value= Shy_t[1])
    @variable(subproblem, 0 <= Xhp[h=1:24] <= maxhp, SDDP.State,initial_value= Shp_t[1])
    
    #Control Variables
    @variable(subproblem, 0 <= turbshift[i=1:nshift,h=1:24] <= Cdr_shift[i] * A_shift_turb[:,"Tech. $(i-1)"][(node-1)*24+h])
    @variable(subproblem, 0 <= pumpshift[i=1:nshift,h=1:24] <= Cdr_shift[i] * A_shift_pump[:,"Tech. $(i-1)"][(node-1)*24+h])

    @variable(subproblem, 0 <= turbshed[i=1:nshed,h=1:24] <= Cdr_shed[i] * A_shed[:, "Tech. $(i-1)"][(node-1)*24+h])    
    
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
    @constraint(subproblem, watershed[i=1:nshed,h=1:24],
    Xshed[i,h].out -(h==1 ? Xshed[i,24].in : Xshed[i,h-1].out) + turbshed[i,h] == 0 )

    @constraint(subproblem, watershiftup[i=1:nshift,h=1:24],
    Xshiftup[i,h].out -(h==1 ? Xshiftup[i,24].in : Xshiftup[i,h-1].out) + turbshift[i,h] - pumpshift[i,h] == 0)
    
    @constraint(subproblem, watershiftdo[i=1:nshift,h=1:24],
    Xshiftdo[i,h].out - (h==1 ? Xshiftdo[i,24].in : Xshiftdo[i,h-1].out) - turbshift[i,h] + pumpshift[i,h] == 0)
    
    @constraint(subproblem, watershiftcont[i=1:nshift,h=1:24],
    Xshiftcont[i,h].out - (h==1 ? Xshiftcont[i,24].in : Xshiftcont[i,h-1].out) + turbshift[i,h] == 0 )
    
    @constraint(
        subproblem, demand_sat[h=1:24],
        0 == - demresi[h] + slackmoins[h] - slackplus[h] + gth_CCGT[h] + gth_TAG[h]+ gnk[h] +
         E * (turbhy[h] + turbhp[h] - pumphp[h]) + imports[h] - exports[h]+
         sum(turbshed[i,h] for i in 1:nshed) +
         sum(turbshift[i,h] - pumpshift[i,h] for i in 1:nshift)
        )

    @constraint(subproblem, waterhy[h=1:24],
    Xhy[h].out == (h==1 ? Xhy[24].in : Xhy[h-1].out) - turbhy[h] + inflow_conv[(node-1)*24+h])

    @constraint(subproblem, waterhp[h=1:24],
    Xhp[h].out == (h==1 ? Xhp[24].in : Xhp[h-1].out) - turbhp[h] + pumphp[h] + inflow_step[(node-1)*24+h])


    # Objective
    @stageobjective(subproblem, 
    sum(Vth_CCGT*gth_CCGT[h] + Vth_TAG*gth_TAG[h] + Vnk*gnk[h] + E*(Vhy*turbhy[h] + Vhp*turbhp[h])+ 
        sum(Vdr_shed[i]*turbshed[i,h] for i in 1:nshed) + V_interco*(imports[h]+exports[h])+
        sum(Vdr_shift[i]*turbshift[i,h] for i in 1:nshift)+
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
            [:imports,:exports,:slackplus,:slackmoins,:Xshed,:Xshiftup,:Xshiftdo,:Xshiftcont,:Xhy,:Xhp,:turbshift,:pumpshift,:turbshed,:gth_CCGT,:gth_TAG,:gnk,:turbhy,:turbhp,:pumphp],
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

for i in 1:nshift
    println("Ecriture niveau et decisions Tech $i Shifting....")
    pumping = Array{Float64}(undef,T*24 ,Nbsimus)
    turbining = Array{Float64}(undef,T*24 ,Nbsimus)
    niveaucont = Array{Float64}(undef,T*24 ,Nbsimus)
    for t in 1:T
        for h in 1:24
            for sim in 1:Nbsimus
                turbining[(t-1)*24+h,sim] = simulations[sim][t][:turbshift][i,h]
                pumping[(t-1)*24+h,sim] = simulations[sim][t][:pumpshift][i,h]
                niveaucont[(t-1)*24+h,sim] = simulations[sim][t][:Xshiftcont][i,h].out
            end
        end
    end
    writedlm(pathoutputs*"turbshiftTech$i.txt",turbining)
    writedlm(pathoutputs*"pumpshiftTech$i.txt",pumping)
    writedlm(pathoutputs*"niveaucontTech$i.txt",niveaucont)
    println("Ecriture niveau et decisions Tech $i Shifting : ok")
end


for i in 1:nshed
    println("Ecriture niveau et decisions Tech $i Shedding....")
    turbining = Array{Float64}(undef,T*24 ,Nbsimus)
    niveau = Array{Float64}(undef,T*24,Nbsimus)
    for t in 1:T
        for h in 1:24
            for sim in 1:Nbsimus
                turbining[(t-1)*24+h,sim] = simulations[sim][t][:turbshed][i,h]
                niveau[(t-1)*24+h,sim] = simulations[sim][t][:Xshed][i,h].out
            end
        end
    end
    writedlm(pathoutputs*"turbshedTech$i.txt",turbining)
    writedlm(pathoutputs*"niveaushedTech$i.txt",niveau)
    println("Ecriture niveau et decisions Tech $i Shedding : ok")
end


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
Imp =  Array{Float64}(undef,T*24 ,Nbsimus)
Exp = Array{Float64}(undef,T*24 ,Nbsimus)
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
writedlm(pathoutputs*"pumphp.txt",Pumphp)
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

println("Obtention wat. values...")
couts_opportshift1 = Array{Float64}(undef,T,Nbsimus)
couts_opportshift2 = Array{Float64}(undef,T,Nbsimus)
couts_opportshift3 = Array{Float64}(undef,T,Nbsimus)
couts_opportshift4 = Array{Float64}(undef,T,Nbsimus)
couts_opportshift5 = Array{Float64}(undef,T,Nbsimus)
couts_opportshift6 = Array{Float64}(undef,T,Nbsimus)
couts_opportshift7 = Array{Float64}(undef,T,Nbsimus)
couts_opportshift8 = Array{Float64}(undef,T,Nbsimus)
couts_opportshift9 = Array{Float64}(undef,T,Nbsimus)
couts_opportshed1 = Array{Float64}(undef,T,Nbsimus)
couts_opportshed2 = Array{Float64}(undef,T,Nbsimus)
couts_opportshed3 = Array{Float64}(undef,T,Nbsimus)
couts_opportshed4 = Array{Float64}(undef,T,Nbsimus)
couts_opporthy = Array{Float64}(undef,T,Nbsimus)
couts_opporthp = Array{Float64}(undef,T,Nbsimus)
for t in 1:T
    local V = SDDP.ValueFunction(model; node=t)
    for sim in 1:Nbsimus
        hydro_pump = Dict("Xhp[$h]" => simulations[sim][t][:Xhp][h].out for h in 1:24)
        hydro_conv = Dict("Xhy[$h]" => simulations[sim][t][:Xhy][h].out for h in 1:24)
        shed = Dict("Xshed[$j,$h]" => simulations[sim][t][:Xshed][j,h].out for h in 1:24 for j in 1:nshed)
        shiftcont = Dict("Xshiftcont[$j,$h]" => simulations[sim][t][:Xshiftcont][j,h].out for h in 1:24 for j in 1:nshift)
        shiftup = Dict("Xshiftup[$j,$h]" => simulations[sim][t][:Xshiftup][j,h].out for h in 1:24 for j in 1:nshift)
        shiftdo = Dict("Xshiftdo[$j,$h]" => simulations[sim][t][:Xshiftdo][j,h].out for h in 1:24 for j in 1:nshift)
        levels = merge(merge(merge(merge(merge(hydro_conv,hydro_pump),shed),shiftcont),shiftdo),shiftup)
        c, valeurs = SDDP.evaluate(V,levels)
        couts_opportshed1[t,sim] = -valeurs[Symbol("Xshed[1,24]")]
        couts_opportshed2[t,sim] = -valeurs[Symbol("Xshed[2,24]")]
        couts_opportshed3[t,sim] = -valeurs[Symbol("Xshed[3,24]")]
        couts_opportshed4[t,sim] = -valeurs[Symbol("Xshed[4,24]")]
        couts_opportshift1[t,sim] = -valeurs[Symbol("Xshiftup[1,24]")]
        couts_opportshift2[t,sim] = -valeurs[Symbol("Xshiftup[2,24]")]
        couts_opportshift3[t,sim] = -valeurs[Symbol("Xshiftup[3,24]")]
        couts_opportshift4[t,sim] = -valeurs[Symbol("Xshiftup[4,24]")]
        couts_opportshift5[t,sim] = -valeurs[Symbol("Xshiftup[5,24]")]
        couts_opportshift6[t,sim] = -valeurs[Symbol("Xshiftup[6,24]")]
        couts_opportshift7[t,sim] = -valeurs[Symbol("Xshiftup[7,24]")]
        couts_opportshift8[t,sim] = -valeurs[Symbol("Xshiftup[8,24]")]
        couts_opportshift9[t,sim] = -valeurs[Symbol("Xshiftup[9,24]")]            
        couts_opporthy[t,sim] = -valeurs[Symbol("Xhy[24]")]
        couts_opporthp[t,sim] = -valeurs[Symbol("Xhp[24]")]
    end
end
println("Fin extraction water value")
writedlm("C:/Users/jujua/Desktop/M2_EEET/Stage CEC/Sorties_remplissage/Sorties_N03//With_DR/margvalshedTech1.txt",couts_opportshed1)
writedlm("C:/Users/jujua/Desktop/M2_EEET/Stage CEC/Sorties_remplissage/Sorties_N03//With_DR/margvalshedTech2.txt",couts_opportshed2)
writedlm(pathoutputs*"margvalshedTech3.txt",couts_opportshed3)
writedlm(pathoutputs*"margvalshedTech4.txt",couts_opportshed4)
writedlm(pathoutputs*"margvalshiftTech1.txt",couts_opportshift1)
writedlm(pathoutputs*"margvalshiftTech2.txt",couts_opportshift2)
writedlm(pathoutputs*"margvalshiftTech3.txt",couts_opportshift3)
writedlm(pathoutputs*"margvalshiftTech4.txt",couts_opportshift4)
writedlm(pathoutputs*"margvalshiftTech5.txt",couts_opportshift5)
writedlm(pathoutputs*"margvalshiftTech6.txt",couts_opportshift6)
writedlm(pathoutputs*"margvalshiftTech7.txt",couts_opportshift7)
writedlm(pathoutputs*"margvalshiftTech8.txt",couts_opportshift8)
writedlm(pathoutputs*"margvalshiftTech9.txt",couts_opportshift9)
writedlm("C:/Users/jujua/Desktop/M2_EEET/Stage CEC/Sorties_remplissage/Sorties_N03//With_DR/margvalhy.txt",couts_opporthy)
writedlm(pathoutputs*"margvalhp.txt",couts_opporthp)
println("Fin extraction water value")