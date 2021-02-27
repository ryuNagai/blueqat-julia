module BackendFunctions
    include("./UndirectedGraph/UndirectedGraph.jl")
    include("./StateVector/StateVector.jl")
    include("./TensorNetworkStates/TensorNetworkStates.jl")
    include("./backend_Models.jl")
    include("../Gate.jl")

    import .UndirectedGraphBackend
    import .StateVectorBackend
    import .TensorNetworkStatesBackend
    using Reexport
    @reexport using .GateSet
    @reexport using .BackendModels

    export execute_backend, get_counts

    function execute_backend(n_qregs::Int, gates::Array{Gate,1}, model::UndirectedGraphModel, args)
        if length(args) > 0
            res = UndirectedGraphBackend.execute_backend(n_qregs::Int, gates, model, args[1])
        else
            res = UndirectedGraphBackend.execute_backend(n_qregs::Int, gates, model, [])
        end
        return res
    end

    function execute_backend(n_qregs::Int, gates::Array{Gate,1}, model::StateVectorModel, args)
        if length(args) > 0
            states, cregs = StateVectorBackend.execute_backend(n_qregs, gates, model, args[1])
        else
            states, cregs = StateVectorBackend.execute_backend(n_qregs, gates, model, 1)
        end
        res = StateVectorResults(states, cregs)
        return res
    end

    function execute_backend(n_qregs::Int, gates::Array{Gate,1}, model::TensorNetworkStatesModel, args)
        if length(args) > 0
            states, info = TensorNetworkStatesBackend.execute_backend(n_qregs, gates, model, args[1])
        else
            states, info = TensorNetworkStatesBackend.execute_backend(n_qregs, gates, model, true)
        end
        res = TensorNetworkStatesResults(states, info)
        return res
    end

    function get_counts(results::StateVectorResults)
        dict = Dict{String,Int64}()
        for creg in results.cregs
            str = ""
            for i in creg
                str *= repr(i)
            end
            try
                dict[str] += 1
            catch error
                dict[str] = 1
            end
        end
        return dict
    end
end