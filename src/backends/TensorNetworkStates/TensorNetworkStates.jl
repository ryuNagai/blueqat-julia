module TensorNetworkStatesBackend

    include("./TensorNetworkStatesFunctions.jl")
    include("../backend_Models.jl")
    export execute_backend
    using .TensorNetworkStatesFunctions
    using .BackendModels

    function execute_backend(n_qregs::Int, gates, model, meas_all::Bool)
        _mps = execute_operation(n_qregs, gates, model)
        output_state = Vector{ComplexF64}(undef, 0)
        for i=0:2^n_qregs-1
            res = restore(_mps, i) # MPS で表された状態から、| bin(i) > の係数を計算する関数
            push!(output_state, res)
        end
        info = mps_size(_mps)
        return (output_state, info)
    end

    function execute_backend(n_qregs::Int, gates, model, measure_basis::Array{Int, 1})
        if length(measure_basis) != n_qregs
            error("Invalid measure basis.")
        end
        _mps = execute_operation(n_qregs, gates, model)
        meas_idx = 0
        for (i, val) in enumerate(measure_basis)
            if val == 1
                meas_idx += 2^(n_qregs - i)
            elseif val > 1
                error("Invalid measure basis.")
            end
        end
        res = restore(_mps, meas_idx)
        info = mps_size(_mps)
        return (res, info)
    end

    function execute_operation(n_qregs::Int, gates, model)
        if model.zero_state == true
            mps = init_zero_MPS(n_qregs)
        else
            if length(model.init_state) != 2^n_qregs
                error("Initial state length must be equal to 2^(n_qubits).")
            else
                mps = init_MPS(model.init_state, model.param)
            end
        end

        backend_gates = call_backend_gates(gates)
        _backend_gates = copy(backend_gates)
        parsed_gates = parse_gates(_backend_gates)
        _mps = copy(mps)

        for gate in parsed_gates
            _mps = apply!(gate, _mps, model.param)
        end
        return _mps
    end


    function call_backend_gates(gates)
        backend_gates = []
        for gate in gates
            name = gate._name
            fields = fieldnames(typeof(gate))
            field_names = []
            for i in fields
                push!(field_names, string(i))
            end
            given_property = [getproperty(gate, property) for property in fields]
            code = gate._name * "("
            for prop in given_property[1:end-1]
                code *= string(prop)
                code *= ", "
            end
            code *= ")"
            new_gate = eval(Meta.parse(code))
            push!(backend_gates, new_gate)
        end
        return backend_gates
    end

    function parse_gates(backend_gates)
        parsed_gates =[]
        for gate in backend_gates
            _gates = gate_parser(gate)
            for _gate in _gates
                push!(parsed_gates, _gate)
            end
        end
        return parsed_gates
    end

end