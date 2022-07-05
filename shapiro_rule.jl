# ======================= Shapiro's Stopping Rule =========================== #
struct Shapiro <: AbstractStoppingRule
    num_replications::Int
    iteration_period::Int
    z_score::Float64
    epsilon::Float64
    verbose::Bool
    function Shapiro(;
        num_replications,
        iteration_period = 1,
        z_score = 1.96,
        epsilon = 100000000,
        verbose = true,
    )
        return new(num_replications, iteration_period, z_score, epsilon, verbose)
    end
end

stopping_rule_status(::Shapiro) = :shapiro

function convergence_test(
    graph::PolicyGraph,
    log::Vector{Log},
    rule::Shapiro,
)
    if length(log) % rule.iteration_period != 0
        # Only run this convergence test every rule.iteration_period iterations.
        return false
    end
    results = simulate(graph, rule.num_replications)
    objectives =
        map(simulation -> sum(s[:stage_objective] for s in simulation), results)
    sample_mean = Statistics.mean(objectives)
    sample_ci =
        rule.z_score * Statistics.std(objectives) / sqrt(rule.num_replications)

    current_bound = log[end].bound
    if rule.verbose
        println(
            "Status of difference: [",
            print_value(sample_mean + sample_ci - current_bound),
            ", ",
            print_value(rule.epsilon),
            "]",
        )
    end
    if graph.objective_sense == MOI.MIN_SENSE
        return sample_mean + sample_ci - current_bound <= rule.epsilon
    elseif graph.objective_sense == MOI.MAX_SENSE
        return  current_bound - (sample_mean - sample_ci) <= rule.epsilon
    else
        # If sense is none of the above for some awkward reason, return to
        # previous criteria
        return sample_mean + sample_ci - rule.epsilon <= current_bound
    end
end