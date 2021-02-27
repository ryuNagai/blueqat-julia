module TensorNetworkStatesFunctions
    include("./TensorNetworkStatesGates.jl")
    include("./MPSforQuantum.jl")
    using DataStructures
    using Reexport
    @reexport using .TensorNetworkStatesGates
    using .MPSforQuantum
    export apply!, restore, init_zero_MPS, gate_parser, mps_size

    function init_zero_MPS(N::Int64)
        mps = []
        for i in 1:N
            push!(mps, [reshape([1.0 + 0.0im], 1, 1), reshape([0.0 + 0.0im], 1, 1)])
        end
        return mps
    end

    function init_MPS(state::Array{ComplexF64,1})
        mps = MPS(state, param, 'l')
        return mps
    end

    # Restore prob. ampl. of a state from MPS
    function restore(arrs::Array{Any, 1}, n::Int64) # n is decimal representation of the state '0110...'
        N = Int64(size(arrs)[1])
        s = bitstring(n)[end - (N - 1):end] # 後ろからN bit目まで
        phys_idx = [convert(Int64, s[i]) - 48 for i = length(s):-1:1] # 後ろが最上位ビット
        return prod([arrs[i][phys_idx[i] + 1] for i = 1:length(phys_idx)])[1]
    end

    function mps_size(arrs::Array{Any, 1})
        d = OrderedDict()
        params = 0
        N = size(arrs)[1]
        for i = 1:N
            #println("array $i's size: ", size(arrs[i][1]))
            d["array $i's size"] = size(arrs[i][1])
            params += length(arrs[i][1]) * 2
        end
        #println("Num of parameters: ", params)
        d["Num of parameters"] = params
        #println("2^N: ", 2^N)
        d["2^N"] = 2^N
        return d
    end

    """
    X gate
    """
    function apply!(gate::X, mps::Array{Any, 1}, param::Number)
        target = gate._target
        mpo = convert(Array{ComplexF64,2}, [0 1; 1 0])
        mps_ = OneQubitGate(mps, mpo, target)
    end

    """
    Y gate
    """
    function apply!(gate::Y, mps::Array{Any, 1}, param::Number)
        target = gate._target
        mpo = convert(Array{ComplexF64,2}, [0 -1im; 1im 0])
        mps_ = OneQubitGate(mps, mpo, target)
    end

    """
    Z gate
    """
    function apply!(gate::Z, mps::Array{Any, 1}, param::Number)
        target = gate._target
        mpo = convert(Array{ComplexF64,2}, [1 0; 0 -1])
        mps_ = OneQubitGate(mps, mpo, target)
    end

    """
    H gate
    """
    function apply!(gate::H, mps::Array{Any, 1}, param::Number)
        target = gate._target
        mpo = convert(Array{ComplexF64,2}, [1 1; 1 -1] ./ sqrt(2))
        mps_ = OneQubitGate(mps, mpo, target)
    end

    """
    S gate
    """
    function apply!(gate::S, mps::Array{Any, 1}, param::Number)
        target = gate._target
        mpo = convert(Array{ComplexF64,2}, [1 0; 0 1im])
        mps_ = OneQubitGate(mps, mpo, target)
    end

    """
    T gate
    """
    function apply!(gate::T, mps::Array{Any, 1}, param::Number)
        target = gate._target
        mpo = convert(Array{ComplexF64,2}, [1 0; 0 _exp(0.25im * pi)])
        mps_ = OneQubitGate(mps, mpo, target)
    end

    """
    RX gate
    """
    function apply!(gate::RX, mps::Array{Any, 1}, param::Number)
        target = gate._target
        angpi = gate._theta * 0.5 / pi
        a = cospi(angpi)
        b = -1im*sinpi(angpi)
        mpo = convert(Array{ComplexF64,2}, [a b; b a])
        mps_ = OneQubitGate(mps, mpo, target)
    end

    """
    RZ gate
    """
    function apply!(gate::RZ, mps::Array{Any, 1}, param::Number)
        target = gate._target
        ang = gate._theta * 0.5
        mpo = convert(Array{ComplexF64,2}, [_exp(-1im*ang) 0; 0 _exp(1im*ang)])
        mps_ = OneQubitGate(mps, mpo, target)
    end

    """
    RY gate
    """
    function apply!(gate::RY, mps::Array{Any, 1}, param::Number)
        target = gate._target
        angpi = gate._theta * 0.5 / pi
        a = cospi(angpi)
        b = sinpi(angpi)
        mpo = convert(Array{ComplexF64,2}, [a -b; b a])
        mps_ = OneQubitGate(mps, mpo, target)
    end

    """
    U3 gate
    """
    function apply!(gate::U3, mps::Array{Any, 1}, param::Number)
        target = gate._target
        theta = gate._theta * 0.5
        thetapi = theta / pi
        phipi = gate._phi / pi
        lambdpi = gate._lambd / pi
        a = cospi(thetapi)
        b = -_exp(1im*lambpi)*sinpi(thetapi)
        c = _exp(1im*phipi)*sinpi(thetapi)
        d = _exp(1im*(phipi+lambpi))*cospi(thetapi)
        mpo = convert(Array{ComplexF64,2}, [a b; c d])
        mps_ = OneQubitGate(mps, mpo, target)
    end

    """
    CX gate
    """
    function apply!(gate::CX, mps::Array{Any, 1}, param::Number)
        control = gate._control
        target = gate._target
        _arr = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
        mpo = convert(Array{ComplexF64,2}, _arr)
        mps_ = TwoQubitGate(mps, mpo, control, target, param)
    end

    """
    CZ gate
    """
    function apply!(gate::CZ, mps::Array{Any, 1}, param::Number)
        control = gate._control
        target = gate._target
        _arr = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]
        mpo = convert(Array{ComplexF64,2}, _arr)
        mps_ = TwoQubitGate(mps, mpo, control, target, param)
    end

    """
    CY gate
    """
    function apply!(gate::CY, mps::Array{Any, 1}, param::Number)
        control = gate._control
        target = gate._target
        _arr = [1 0 0 0; 0 1 0 0; 0 0 0 -1im; 0 0 1im 0]
        arr = convert(Array{ComplexF64,2}, _arr)
        mps_ = TwoQubitGate(mps, arr, control, target, param)
    end

    """
    SWAP gate
    """
    function apply!(gate::SWAP, mps::Array{Any, 1}, param::Number)
        control = gate._control
        target = gate._target
        _arr = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
        arr = convert(Array{ComplexF64,2}, _arr)
        mps_ = TwoQubitGate(mps, arr, control, target, param)
    end

    """
    CRX gate
    """
    function apply!(gate::CRX, mps::Array{Any, 1}, param::Number)
        control = gate._control
        target = gate._target
        ang = gate._theta * 0.5
        angpi = ang / pi
        a = cospi(angpi)
        b = -1im*sinpi(angpi)
        _arr = [1 0 0 0; 0 1 0 0; 0 0 a b; 0 0 b a]
        arr = convert(Array{ComplexF64,2}, _arr)
        mps_ = TwoQubitGate(mps, arr, control, target, param)
    end

    """
    CRY gate
    """
    function apply!(gate::CRY, mps::Array{Any, 1}, param::Number)
        control = gate._control
        target = gate._target
        ang = gate._theta * 0.5
        angpi = ang / pi
        a = cospi(angpi)
        b = sinpi(angpi)
        _arr = [1 0 0 0; 0 1 0 0; 0 0 a -b; 0 0 b a]
        arr = convert(Array{ComplexF64,2}, _arr)
        mps_ = TwoQubitGate(mps, arr, control, target, param)
    end

    """
    CRZ gate
    """
    function apply!(gate::CRZ, mps::Array{Any, 1}, param::Number)
        control = gate._control
        target = gate._target
        ang = gate._theta * 0.5
        _arr = [1 0 0 0; 0 1 0 0; 0 0 _exp(-1im*ang) 0; 0 0 0 _exp(1im*ang)]
        arr = convert(Array{ComplexF64,2}, _arr)
        mps_ = TwoQubitGate(mps, arr, control, target, param)
    end

    """
    CP gate
    """
    function apply!(gate::CP, mps::Array{Any, 1}, param::Number)
        control = gate._control
        target = gate._target
        ang = gate._theta
        _arr = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 _exp(1im*ang)]
        arr = convert(Array{ComplexF64,2}, _arr)
        mps_ = TwoQubitGate(mps, arr, control, target, param)
    end

    function _exp(theta::ComplexF64)
        thetapi = theta / pi / im
        if thetapi == conj(thetapi)
            thetapi = convert(Float64, thetapi)
        else
            error("Invalid gate parameters.")
        end
        res = cospi(thetapi) + im * sinpi(thetapi)
        return res
    end

    function gate_parser(gate::T) where T <: TNS_TwoQubitGate
        c = gate._control
        t = gate._target
        swap_idx = []
        if t > c + 1
            for i in 1:(t - c - 1)
                push!(swap_idx, (c+i-1, c+i))
            end
            gate._control = t - 1
        elseif c > t + 1
            for i in 1:(c - t - 1)
                push!(swap_idx, (t+i-1, t+i))
            end
            gate._target = c - 1
        end
        parsed_gates = []
        for swap in swap_idx
            push!(parsed_gates, SWAP(swap[1], swap[2]))
        end
        push!(parsed_gates, gate)
        for swap in reverse(swap_idx)
            push!(parsed_gates, SWAP(swap[1], swap[2]))
        end
        return parsed_gates
    end

    function gate_parser(gate::T) where T <: TNS_OneQubitGate
        return [gate]
    end

end
