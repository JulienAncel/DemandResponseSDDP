# DemandResponseSDDP

This project contains :

* Folder France2050/, with code and data related to the paper "Assessing the potential of demand response as a source of flexibility in low-carbon power systems: insights from the French case", J. Ancel and O. Massol, Jun. 2022
      
* Folder NWE2023/, with code and data related to the paper "Stochastic Dual Dynamic Programming as a modelling Tool for Power Systems with Demand Response and intermittent Renewables", J. Ancel and O. Massol, Jul. 2022.

As described in those papers, the stopping rule proposed by Shapiro in "Analysis of stochastic dual dynamic programming method, A. Shapiro, European Journal of Operational Research (2011)" is used. The rule is not implemented in SDDP.jl. So, the user should copy the code of the file shapiro_rule.jl into their own version of the Julia SDDP package, and more precisely in the file .julia/packages/SDDP/g2xzQ/src/plugins/stopping_rules.jl . A refresh of the module is required before first use.

In order to use postreatment files i.e. NWE2023/postreat.py and France2050/postraitement.py, outputs folders, output files and paths must be created and written accordingly.

It requires : 

* Julia with packages SDDP.jl, CPLEX.jl (which requires the CPLEX solver)

* Python with packages numpy, matplotlib.pyplot and pandas.
