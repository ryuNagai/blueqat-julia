module BackendModels
    using DataStructures
    export Device, Results
    export UndirectedGraphModel, StateVectorModel, TensorNetworkStatesModel
    export StateVectorResults, TensorNetworkStatesResults

    abstract type Device end
    abstract type Results end

    struct UndirectedGraphModel <: Device
    end

    function UndirectedGraphModel(x)
        return UndirectedGraphModel()
    end

    struct StateVectorModel <: Device
        zero_state::Bool
        init_state::Array{ComplexF64, 1}
    end

    function StateVectorModel(init_state)
        if length(init_state) == 0
            return StateVectorModel(true, [0im])
        else
            return StateVectorModel(false, init_state)
        end
    end

    struct StateVectorResults <: Results
        states::Array{Any, 1}
        cregs::Array{Any, 1}
    end

    struct TensorNetworkStatesModel <: Device
        zero_state::Bool
        init_state::Array{ComplexF64, 1}
        param::Number
    end

    function TensorNetworkStatesModel(x)
        if length(x) == 1
            return TensorNetworkStatesModel(true, [0im], x[1])
        else
            return TensorNetworkStatesModel(false, x[2], x[1])
        end
    end

    struct TensorNetworkStatesResults <: Results
        states::Array{ComplexF64, 1}
        info::OrderedDict{Any,Any}
    end

end