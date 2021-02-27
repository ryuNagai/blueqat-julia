
module TensorNetworkStatesGates

    export OneQubitGateTensor, TwoQubitGateTensor, TNS_Gate, TNS_OneQubitGate, TNS_TwoQubitGate, GateTensor
    abstract type GateTensor end
    abstract type TNS_Gate end
    abstract type TNS_OneQubitGate <: TNS_Gate end
    abstract type TNS_TwoQubitGate <: TNS_Gate end

    mutable struct OneQubitGateTensor <: GateTensor
        item::Array{ComplexF64, 2}
        target::Int
    end

    mutable struct TwoQubitGateTensor <: GateTensor
        item::Array{ComplexF64, 2}
        control::Int
        target::Int
    end

    mutable struct X <: TNS_OneQubitGate
        _target::Int64
    end

    mutable struct Y <: TNS_OneQubitGate
        _target::Int64
    end

    mutable struct Z <: TNS_OneQubitGate
        _target::Int64
    end

    mutable struct H <: TNS_OneQubitGate
        _target::Int64
    end

    mutable struct T <: TNS_OneQubitGate
        _target::Int64
    end

    mutable struct S <: TNS_OneQubitGate
        _target::Int64
    end

    mutable struct RX <: TNS_OneQubitGate
        _target::Int64
        _theta::Float64
    end

    mutable struct RY <: TNS_OneQubitGate
        _target::Int64
        _theta::Float64
    end

    mutable struct RZ <: TNS_OneQubitGate
        _target::Int64
        _theta::Float64
    end

    mutable struct U3 <: TNS_OneQubitGate
        _target::Int64
        _theta::Float64
        _phi::Float64
        _lambd::Float64
    end

    mutable struct CZ <: TNS_TwoQubitGate
        _control::Int64
        _target::Int64
    end

    mutable struct CX <: TNS_TwoQubitGate
        _control::Int64
        _target::Int64
    end

    mutable struct CY <: TNS_TwoQubitGate
        _control::Int64
        _target::Int64
    end

    mutable struct SWAP <: TNS_TwoQubitGate
        _control::Int64
        _target::Int64
    end

    mutable struct CP <: TNS_TwoQubitGate
        _control::Int64
        _target::Int64
        _theta::Float64
    end

    mutable struct CRX <: TNS_TwoQubitGate
        _control::Int64
        _target::Int64
        _theta::Float64
    end

    mutable struct CRY <: TNS_TwoQubitGate
        _control::Int64
        _target::Int64
        _theta::Float64
    end

    mutable struct CRZ <: TNS_TwoQubitGate
        _control::Int64
        _target::Int64
        _theta::Float64
    end

    # export all UG_Gate and TwoQubitGate
    for n in names(@__MODULE__; all=true)
        if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
            if @eval typeof($n) <: DataType
                if @eval $n <: TNS_Gate
                    @eval export $n
                end
            end
        end
    end
end
