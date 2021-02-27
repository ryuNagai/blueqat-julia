module MPSforQuantum
    export MPS, restore, OneQubitGate, TwoQubitGate, dstack, ddstack, init_rand_MPS
    export CX, inner_product, expectation, SVD_R, SVD_L, iterative_ground_state_search, j1j2_2D_Hamiltonian_MPO
    using LinearAlgebra
    using TensorOperations

    function SVD(C::Array{ComplexF64,2}, eps::Float64)
        #eps = 1e-2
        A = svd(C)
        filter!((x) -> x > eps, A.S)
        l = length(A.S)
        return A.U[:, 1:l], diagm(A.S), A.Vt[1:l, :]
        #return A.U[:, 1:l], diagm(A.S), transpose(A.Vt)[1:l, :]
    end

    function SVD(C::Array{ComplexF64,2}, D::Int64)
        A = svd(C)
        if length(A.S) > D
            return A.U[:, 1:D], diagm(A.S[1:D]), A.Vt[1:D, :]
            #return A.U[:, 1:D], diagm(A.S[1:D]), transpose(A.Vt)[1:D, :]
        else
            return A.U[:, :], diagm(A.S), A.Vt[:, :]
            #return A.U[:, :], diagm(A.S), transpose(A.Vt)[:, :]
        end
    end

    function SVD_L(C::Array{ComplexF64,2}, eps::Float64)
        U, S, Vt = SVD(C, eps)
        return U, S * Vt
    end

    function SVD_L(C::Array{ComplexF64,2}, D::Int64)
        U, S, Vt = SVD(C, D)
        return U, S * Vt
    end

    function SVD_R(C::Array{ComplexF64,2}, eps::Float64)
        U, S, Vt = SVD(C, eps)
        return U * S, Vt
    end

    function SVD_R(C::Array{ComplexF64,2}, D::Int64)
        U, S, Vt = SVD(C, D)
        return U * S, Vt
    end

    function MPS(C::Array{ComplexF64,1}, param, normalize::Char='l')
        d = 2
        N = Int64(log2(size(C)[1]))
        arrs = []
        r = 1
        if normalize == 'l'
            for i = 1:N-1
                C = reshape(C, d * r, d^(N - i))
                tmp_A, C = SVD_L(C, param)
                r = size(tmp_A, 2)
                col = convert(Int64, size(tmp_A, 1) / 2)
                push!(arrs, [tmp_A[1:col, :], tmp_A[(col+1):col*2, :]])
            end

            Ct = transpose(C)
            col2 = convert(Int64, size(C, 2) / 2)
            push!(arrs, [C[:, 1:col2], C[:, (col2+1):col2*2]])
            return arrs
        elseif normalize == 'r'
            for i = 1:N-1
                #C = reshape(C, d^(N - i), d * r)
                l = convert(Int64, size(C, 1) / 2)
                C = cat(C[1:l, :], C[l+1:l*2, :], dims=2)
                C, tmp_B = SVD_R(C, param)
                r = size(tmp_B, 1)
                col = convert(Int64, size(tmp_B, 2) / 2)
                push!(arrs, [tmp_B[:, 1:col], tmp_B[:, (col+1):col*2]])
            end

            Ct = transpose(C)
            col2 = convert(Int64, size(C, 1) / 2)
            push!(arrs, [C[1:col2, :], C[(col2+1):col2*2, :]])
            arrs_ = []
            for i = 1:N
                push!(arrs_, arrs[N - (i-1)])
            end
            return arrs_
        end
    end

    function OneQubitGate(arrs::Array{Any, 1}, O::Array{Complex{Float64},2}, n::Int64)
        arrs_ = similar(arrs[n])
        arrs__ = copy(arrs)
        arrs_[1] = arrs[n][1] * O[1, 1] + arrs[n][2] * O[1, 2]
        arrs_[2] = arrs[n][1] * O[2, 1] + arrs[n][2] * O[2, 2]
        arrs__[n] = arrs_
        return arrs__
    end

    function swap_ct(arr::Array{Complex{Float64},2})
        swap = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
        _arr = copy(arr)
        _arr[:, 2] = arr[:, 3]
        _arr[:, 3] = arr[:, 2]
        return swap * _arr
    end

    function reduce(mps::Array{Any, 1}, t1::Int64, t2::Int64, param::Number)
        eps = 1e-3
        arr1 = cat(mps[t1][1], mps[t1][2], dims = 1)
        arr2 = cat(mps[t2][1], mps[t2][2], dims = 2)
        arr = arr1 * arr2
        
        # U, S, V = svd(arr)
        # filter!((x) -> x > eps, S)
        # sdim = length(S)
        U, S, V = SVD(arr, param)
    
        #U1 = U[:, 1:sdim]
        U1 = U
        U1 = [U1[1:convert(Int, size(U1)[1]/2), :], U1[convert(Int, size(U1)[1]/2)+1:size(U1)[1], :]]
        #U2 = diagm(S) * transpose(V)[1:sdim, :]
        U2 = S * V
        U2 = [U2[:, 1:convert(Int, size(U2)[2]/2)], U2[:, convert(Int, size(U2)[2]/2)+1:size(U2)[2]]]
        
        mps[t1] = U1
        mps[t2] = U2
        return mps
    end

    function TwoQubitGate(mps::Array{Any, 1}, arr::Array{Complex{Float64},2}, c::Int64, t::Int64, param::Number)
        eps = 1e-6 # constant cutoff for TwoQubitGate decomposition

        if c == t + 1
            arr = swap_ct(arr)
            (c, t) = (t, c)
        end
        
        arr_00 = transpose(reshape(arr[:, 1], 2, 2))
        arr_01 = transpose(reshape(arr[:, 2], 2, 2))
        arr_10 = transpose(reshape(arr[:, 3], 2, 2))
        arr_11 = transpose(reshape(arr[:, 4], 2, 2))
    
        tensor1 = cat(arr_00, arr_01, dims = 2)
        tensor2 = cat(arr_10, arr_11, dims = 2)
        tensor = cat(tensor1, tensor2, dims = 1)
        
        U, S, V = svd(tensor)
        filter!((x) -> x > eps, S)
        sdim = length(S)
        U1 = U[:, 1:sdim]
        U1 = [U1[1:convert(Int, size(U1)[1]/2), :], U1[convert(Int, size(U1)[1]/2)+1:size(U1)[1], :]]
        U2 = diagm(S) * transpose(V)[1:sdim, :]
        U2 = [U2[:, 1:convert(Int, size(U2)[2]/2)], U2[:, convert(Int, size(U2)[2]/2)+1:size(U2)[2]]]
        
        after_c = kron(U1[1], mps[c][1]) + kron(U1[2], mps[c][2])
        after_t = kron(U2[1], mps[t][1]) + kron(U2[2], mps[t][2])
        c_row = convert(Int, size(after_c)[1]/2)
        t_col = convert(Int, size(after_t)[2]/2)
        
        mps[t][1] = convert(Array{ComplexF64,2}, after_t[:, 1:t_col])
        mps[t][2] = convert(Array{ComplexF64,2}, after_t[:, t_col+1:size(after_t)[2]])
        mps[c][1] = convert(Array{ComplexF64,2}, after_c[1:c_row, :])
        mps[c][2] = convert(Array{ComplexF64,2}, after_c[c_row+1:size(after_c)[1], :])

        mps = reduce(mps, c, t, param)

        return mps
    end

    function dstack(operators)
        return cat(operators..., dims = 3)
    end
    
    function ddstack(operators)
        m = size(operators)[1]
        tmp2 = []
        for i=1:m
            push!(tmp2, cat(operators[i]..., dims=4))
        end
        tmp3 = Tuple(tmp2)
        mpo = cat(tmp3..., dims=3)
        return mpo
    end

    function inner_product(arrs1::Array{Any, 1}, arrs2::Array{Any, 1})
        N = Int64(size(arrs1)[1])
        ip = arrs1[1][1]' * arrs2[1][1] + arrs1[1][2]' * arrs2[1][2]
        for i = 2:N
            phys0 = arrs1[i][1]' * ip * arrs2[i][1]
            phys1 = arrs1[i][2]' * ip * arrs2[i][2]
            ip = phys0 + phys1
        end
        return ip
    end

    # function CX(arrs::Array{Any, 1}, n::Int64, t::Number)
    #     tmp = copy(arrs)
    #     arrs_ = [tmp[n + 1][1]*tmp[n + 2][1] tmp[n + 1][2]*tmp[n + 2][2]
    #             tmp[n + 1][1]*tmp[n + 2][2] tmp[n + 1][2]*tmp[n + 2][1]]
    #     arrs__ = MPSforQuantum.SVD_L(arrs_, t)
    #     col = convert(Int64, size(arrs__[1], 1) / 2)
        
    #     tmp[n + 1][1] = arrs__[1][1:col, :]
    #     tmp[n + 1][2] = arrs__[1][(col+1):(2*col), :]
    #     tmp[n + 2][1] = arrs__[2][:, 1:col]
    #     tmp[n + 2][2] = arrs__[2][:, (col+1):(2*col)]
    
    #     return tmp
    # end

    function pauli()
        pauliX = convert(Array{ComplexF64,2}, [0 1; 1 0])
        pauliY = convert(Array{ComplexF64,2}, [0 -1*im; 1*im 0])
        pauliZ = convert(Array{ComplexF64,2}, [1 0; 0 -1])
        pauliI = convert(Array{ComplexF64,2}, [1 0; 0 1])
        return (pauliX, pauliY, pauliZ, pauliI)
    end

    function pauli_zero()
        zero = convert(Array{ComplexF64,2}, [0 0; 0 0])
        return zero
    end

    function expectation(mps::Array{Any,1}, O::Array{Any,1})
        N_site = size(mps)[1]
        contracted_sites = []
        n_phys = 2 #ss
        for i=1:N_site
            if i==1
                # first step
                tmp_mps = cat(mps[i][1][:], mps[i][2][:], dims=2)
                tmp_mps_dag = cat(mps[i][1]'[:], mps[i][2]'[:], dims=2)
                tmp_O = O[i]
                a_len = size(mps[i][1])[2]
                b_len = size(O[i][1, 1, :])[1] # bb0
                arr_0 = zeros(ComplexF64, n_phys, b_len, a_len)
                arr = zeros(ComplexF64, a_len, b_len, a_len)
                @tensor begin
                    arr_0[sigma1, b, a_] = tmp_O[sigma1, sigma2, b] * tmp_mps[a_, sigma2]
                    arr[a, b, a_] = tmp_mps_dag[a, sigma1] * arr_0[sigma1, b, a_]
                end

            elseif i==N_site
                # last step
                tmp_mps = cat(mps[i][1][:], mps[i][2][:], dims=2)
                tmp_mps_dag = cat(mps[i][1]'[:], mps[i][2]'[:], dims=2)
                tmp_O = O[i]
                tmp_site = contracted_sites[i-1]
                a_len1 = size(mps[i-1][1])[2]
                b_len2 = size(O[i][1, 1, :])[1]
                arr_0 = zeros(ComplexF64, n_phys, a_len1, b_len2)
                arr_1 = zeros(ComplexF64, n_phys, a_len1)
                @tensor begin
                    arr_0[sigma2, a1, b2] = tmp_site[a1, b2, a1_0] * tmp_mps[a1_0, sigma2]
                    arr_1[sigma1, a1] = tmp_O[sigma1, sigma2, b2] * arr_0[sigma2, a1, b2]
                    arr = tmp_mps_dag[a1, sigma1] * arr_1[sigma1, a1]
                end

            else
                # Middle step
                tmp_mps = cat(mps[i][1], mps[i][2], dims=3)
                tmp_mps_dag = cat(mps[i][1]', mps[i][2]', dims=3)
                tmp_O = O[i]
                tmp_site = contracted_sites[i-1]
                a_len1 = size(mps[i-1][1])[2]
                a_len2 = size(mps[i][1])[2]
                b_len1, b_len2 = size(O[i][1, 1, :, :])
                arr_0 = zeros(ComplexF64, n_phys, a_len1, b_len1, a_len2)
                arr_1 = zeros(ComplexF64, n_phys, b_len2, a_len1, a_len2)
                arr = zeros(ComplexF64, a_len2, b_len2, a_len2)
                @tensor begin
                    arr_0[sigma2, a1, b1, a2] = tmp_site[a1, b1, a0_0] * tmp_mps[a0_0, a2, sigma2]
                    arr_1[sigma1, b2, a1, a2] = tmp_O[sigma1, sigma2, b1, b2] * arr_0[sigma2, a1, b1, a2]
                    arr[a2, b2, a2_] = tmp_mps_dag[a2, a1, sigma1] * arr_1[sigma1, b2, a1, a2_]
                end
            end
            push!(contracted_sites, arr)
        end
        return contracted_sites[N_site]
    end

    function R_expression(mps::Array{Any,1}, O::Array{Any,1}, t::Int64)
        # t: target site, sites t+1 to N are merged to R expression
        N_site = size(mps)[1]
        n_phys = 2
        contracted_sites = []
        for i = t+1:N_site
            if i==t+1
                if i==N_site
                    tmp_mps = cat(mps[i][1][:], mps[i][2][:], dims=2)
                    tmp_mps_dag = cat(mps[i][1]'[:], mps[i][2]'[:], dims=2)
                    tmp_O = O[i]
                    a_len1 = size(mps[i-1][1])[2]
                    a_len2 = size(mps[i][1])[2]
                    b_len = size(O[i][1, 1, :])[1]
                    arr = zeros(ComplexF64, a_len1, b_len, a_len1)
                    @tensor begin
                        arr[a0, b0, a0_] = tmp_mps_dag[a0, sigma1] * tmp_O[sigma1, sigma2, b0] * tmp_mps[a0_, sigma2]
                    end
                    push!(contracted_sites, arr)
                else
                    tmp_mps = cat(mps[i][1], mps[i][2], dims=3)
                    tmp_mps_dag = cat(mps[i][1]', mps[i][2]', dims=3)
                    tmp_O = O[i]
                    a_len1 = size(mps[i-1][1])[2]
                    a_len2 = size(mps[i][1])[2]
                    b_len = size(O[i][1, 1, :, :])[1]
                    arr = zeros(ComplexF64, a_len1, a_len2, b_len, b_len, a_len1, a_len2)
                    @tensor begin
                        arr[a0, a1, b0, b1, a0_, a1_] = tmp_mps_dag[a1, a0, sigma1] * tmp_O[sigma1, sigma2, b0, b1] * tmp_mps[a0_, a1_, sigma2]
                    end
                    push!(contracted_sites, arr)
                end
                
            elseif i==N_site
                # last step
                tmp_mps = cat(mps[i][1][:], mps[i][2][:], dims=2)
                tmp_mps_dag = cat(mps[i][1]'[:], mps[i][2]'[:], dims=2)
                tmp_O = O[i]
                tmp_site = contracted_sites[1]
                a_len0 = size(tmp_site)[1]
                a_len1 = size(mps[i-1][1])[2]
                a_len2 = size(mps[i][1])[2]
                b_len = size(O[i][1, 1, :])[1]
                arr_0 = zeros(ComplexF64, n_phys, a_len0, a_len1, b_len, b_len, a_len0)
                arr_1 = zeros(ComplexF64, n_phys, n_phys, a_len0, b_len, b_len, a_len0)
                arr = zeros(ComplexF64, a_len0, b_len, a_len0)
                @tensor begin
                    arr_0[sigma2, a0, a1, b0, b1, a0_] = tmp_site[a0, a1, b0, b1, a0_, a1_] * tmp_mps[a1_, sigma2]
                    arr_1[sigma1, sigma2, a0, b0, b1, a0_] = tmp_mps_dag[a1, sigma1] * arr_0[sigma2, a0, a1, b0, b1, a0_]
                    arr[a0, b0, a0_] = arr_1[sigma1, sigma2, a0, b0, b1, a0_] * tmp_O[sigma1, sigma2, b1]
                end
                contracted_sites[1] = arr
            else
                tmp_mps = cat(mps[i][1], mps[i][2], dims=3)
                tmp_mps_dag = cat(mps[i][1]', mps[i][2]', dims=3)
                tmp_O = O[i]
                tmp_site = contracted_sites[1]
                a_len0 = size(tmp_site)[1]
                a_len1 = size(mps[i-1][1])[2]
                a_len2 = size(mps[i][1])[2]
                b_len = size(O[i][1, 1, :, :])[1]
                arr_0 = zeros(ComplexF64, n_phys, a_len0, a_len1, b_len, b_len, a_len0, a_len2)
                arr_1 = zeros(ComplexF64, n_phys, n_phys, a_len0, a_len2, b_len, b_len, a_len0, a_len2)
                arr = zeros(ComplexF64, a_len0, a_len2, b_len, b_len, a_len0, a_len2)
                @tensor begin
                    arr_0[sigma2, a0, a1, b0, b1, a0_, a2_] = tmp_site[a0, a1, b0, b1, a0_, a1_] * tmp_mps[a1_, a2_, sigma2]
                    arr_1[sigma1, sigma2, a0, a2, b0, b1, a0_, a2_] = tmp_mps_dag[a2, a1, sigma1] * arr_0[sigma2, a0, a1, b0, b1, a0_, a2_]
                    arr[a0, a2, b0, b2, a0_, a2_] = arr_1[sigma1, sigma2, a0, a2, b0, b1, a0_, a2_] * tmp_O[sigma1, sigma2, b1, b2]
                end
                contracted_sites[1] = arr
            end
        end
        return contracted_sites[1]
    end

    function L_expression(mps::Array{Any,1}, O::Array{Any,1}, t::Int64)
        # t: target site, sites 1 to t-1 are merged to L expression
        N_site = size(mps)[1]
        n_phys = 2
        contracted_sites = []
        for i = 1:t-1
            if i==1
                tmp_mps = cat(mps[i][1][:], mps[i][2][:], dims=2)
                tmp_mps_dag = cat(mps[i][1]'[:], mps[i][2]'[:], dims=2)
                tmp_O = O[i]
                a_len = size(mps[i][1])[2]
                b_len = size(O[i][1, 1, :, :])[1]
                arr = zeros(ComplexF64, a_len, b_len, a_len)
                @tensor begin
                    arr[a0, b0, a0_] = tmp_mps_dag[a0, sigma1] * tmp_O[sigma1, sigma2, b0] * tmp_mps[a0_, sigma2]
                end
                push!(contracted_sites, arr)
            else
                tmp_mps = cat(mps[i][1], mps[i][2], dims=3)
                tmp_mps_dag = cat(mps[i][1]', mps[i][2]', dims=3)
                tmp_O = O[i]
                tmp_site = contracted_sites[1]
                a_len0 = size(tmp_site)[1]
                a_len1 = size(mps[i-1][1])[2]
                a_len2 = size(mps[i][1])[2]
                b_len = size(O[i][1, 1, :, :])[1]
                arr_0 = zeros(ComplexF64, n_phys, a_len1, b_len, a_len2)
                arr_1 = zeros(ComplexF64, n_phys, n_phys, a_len2, b_len, a_len2)
                arr = zeros(ComplexF64, a_len2, b_len, a_len2)
                @tensor begin
                    arr_0[sigma2, a1, b1, a2_] = tmp_site[a1, b1, a1_] * tmp_mps[a1_, a2_, sigma2]
                    arr_1[sigma1, sigma2, a2, b1, a2_] = tmp_mps_dag[a2, a1, sigma1] * arr_0[sigma2, a1, b1, a2_]
                    arr[a2, b2, a2_] = arr_1[sigma1, sigma2, a2, b1, a2_] * tmp_O[sigma1, sigma2, b1, b2]
                end
                contracted_sites[1] = arr
            end
        end
        return contracted_sites[1]
    end

    function left_norm_for_2_sites(mps::Array{Any,1}, t::Int64, D::Int64)
        # t: target, site to be left normalized
        mps_ = copy(mps)
        site_1 = cat(mps[t][1], mps[t][2], dims=1)
        site_2 = cat(mps[t+1][1], mps[t+1][2], dims=2)
        mixed_site = site_1 * site_2
        A, M = SVD_L(mixed_site, D)
        col = convert(Int64, size(A, 1) / 2)
        mps_[t] = [A[1:col, :], A[(col+1):col*2, :]]
        col2 = convert(Int64, size(M, 2) / 2)
        mps_[t+1] = [M[:, 1:col2], M[:, (col2+1):col2*2]]
        return mps_
    end

    function right_norm_for_2_sites(mps::Array{Any,1}, t::Int64, D::Int64)
        # t: target, site to be right normalized
        mps_ = copy(mps)
        site_1 = cat(mps[t-1][1], mps[t-1][2], dims=1)
        site_2 = cat(mps[t][1], mps[t][2], dims=2)
        mixed_site = site_1 * site_2
        M, B = SVD_R(mixed_site, D)
        col = convert(Int64, size(M, 1) / 2)
        mps_[t-1] = [M[1:col, :], M[(col+1):col*2, :]]
        col2 = convert(Int64, size(B, 2) / 2)
        mps_[t] = [B[:, 1:col2], B[:, (col2+1):col2*2]]
        return mps_
    end

    function left_most_site_update(mps_::Array{Any,1}, O::Array{Any,1}, t::Int64)
        R = R_expression(mps_, O, t)
        tmp_O = O[t]
        H = zeros(ComplexF64, size(R, 1), size(R, 3), size(tmp_O, 1), size(tmp_O, 2))
        @tensor begin
            H[sigma1, a0, sigma2, a0_] = tmp_O[sigma1, sigma2, b0] * R[a0, b0, a0_]
        end
        H_ = zeros(ComplexF64, size(tmp_O, 1) * size(R, 1), size(tmp_O, 2) * size(R, 3))
        for i=1:size(H, 1), j=1:size(H, 2), k=1:size(H, 3), l=1:size(H, 4)
            H_[(i-1)*size(H, 2) + j, (k-1)*size(H, 4) + l] = H[i,j,k,l]
        end
        v = eigvecs(H_)[:, 1]
        d = convert(Int64, size(v, 1) / 2)
        mps_[t][1][:] = v[1:d]
        mps_[t][2][:] = v[d+1:2*d]
        return mps_
    end
    
    function mid_site_update(mps_::Array{Any,1}, O::Array{Any,1}, t::Int64)
        R = R_expression(mps_, O, t)
        L = L_expression(mps_, O, t)
        tmp_O = O[t]
        H = zeros(ComplexF64, size(tmp_O, 1), size(L, 1), size(R, 1), size(tmp_O, 2), size(L, 3), size(R, 3))
        @tensor begin
            H[sigma1, a0, a1, sigma2, a0_, a1_] = L[a0, b0, a0_] * tmp_O[sigma1, sigma2, b0, b1] * R[a1, b1, a1_]
        end
        H_ = zeros(ComplexF64, size(tmp_O, 1) * size(L, 1) * size(R, 1), size(tmp_O, 2) * size(L, 3) * size(R, 3))
        for i=1:size(H, 1), j=1:size(H, 2), k=1:size(H, 3), l=1:size(H, 4), m=1:size(H, 5), n=1:size(H, 6)
            H_[(i-1)*size(H, 2)*size(H, 3) + (j-1)*size(H, 3) + k, (l-1)*size(H, 5)*size(H, 6) + (m-1)*size(H, 6) + n] = H[i,j,k,l,m,n]
        end
        v = eigvecs(H_)[:, 1]
        d = convert(Int64, size(v, 1) / 2)
        M_1 = transpose(reshape(v[1:d], size(transpose(mps_[t][1]))))
        M_2 = transpose(reshape(v[d+1:2*d], size(transpose(mps_[t][1]))))
        mps_[t][1][:, :] = M_1
        mps_[t][2][:, :] = M_2
        return mps_
    end
        
    function right_most_site_update(mps_::Array{Any,1}, O::Array{Any,1}, t::Int64)
        L = L_expression(mps_, O, t)
        tmp_O = O[t]
        H = zeros(ComplexF64, size(L, 1), size(L, 3), size(tmp_O, 1), size(tmp_O, 2))
        @tensor begin
            H[sigma1, a0, sigma2, a0_] = tmp_O[sigma1, sigma2, b0] * L[a0, b0, a0_]
        end
        H_ = zeros(ComplexF64, size(tmp_O, 1) * size(L, 1), size(tmp_O, 2) * size(L, 3))
        for i=1:size(H, 1), j=1:size(H, 2), k=1:size(H, 3), l=1:size(H, 4)
            H_[(i-1)*size(H, 2) + j, (k-1)*size(H, 4) + l] = H[i,j,k,l]
        end
        v = eigvecs(H_)[:, 1]
        d = convert(Int64, size(v, 1) / 2)
        mps_[t][1][:] = v[1:d]
        mps_[t][2][:] = v[d+1:2*d]
        return mps_
    end

    function iterative_ground_state_search(mps::Array{Any,1}, O::Array{Any,1}, D::Int64, ite::Int64)
        hist = []
        N = size(mps, 1)
        mps_ = copy(mps)
        push!(hist, expectation(mps_, O))
        for i = 1:ite
            for t = 1:N
                if t == 1
                    mps_ = left_most_site_update(mps_, O, t)
                    mps_ = left_norm_for_2_sites(mps_, t, D)  
                elseif t == N
                    mps_ = right_most_site_update(mps_, O, t)
                else
                    mps_ = mid_site_update(mps_, O, t)
                    mps_ = left_norm_for_2_sites(mps_, t, D)
                end
                # push!(hist, expectation(mps_, O))
            end
            
            for t = N-1:-1:1
                mps_ = right_norm_for_2_sites(mps_, t+1, D)
                if t == 1
                    mps_ = left_most_site_update(mps_, O, t)
                else
                    mps_ = mid_site_update(mps_, O, t)
                end  
                push!(hist, expectation(mps_, O))
            end
        end
        return (mps_, hist)
    end

    function init_rand_MPS(N::Int64, D::Int64, normalize::Char='l')
        mps = []
        mps2 = []
        half_N = convert(Int64, floor(N/2))
        for i in 1:half_N
            l = 2^(i-1)
            mps_1 = []
            mps_2 = []
            for j in 1:2
                arr1 = rand(ComplexF64, (min(l, D), min(l*2, D))) / min(l*2, D)
                arr2 = rand(ComplexF64, (min(l*2, D), min(l, D))) / min(l, D)
                push!(mps_1, arr1)
                push!(mps_2, arr2)
            end
            push!(mps, mps_1)
            push!(mps2, mps_2)
        end
        
        if N%2 == 1
            mps_ = []
            for j in 1:2
                arr = rand(ComplexF64, (min(2^half_N, D), min(2^half_N, D))) / min(2^half_N, D)
                push!(mps_, arr)
            end
            push!(mps, mps_)
        end
        
        for j in 1:half_N
            push!(mps, mps2[half_N - (j-1)])
        end
        
        for i in 1:N-1
            mps = left_norm_for_2_sites(mps, i, D)
        end
        norm = mps[N][1]' * mps[N][1] + mps[N][2]' * mps[N][2]
        mps[N][1] = mps[N][1]/sqrt(norm)
        mps[N][2] = mps[N][2]/sqrt(norm)
        if normalize == 'r'
            for i in N:-1:2
                mps = right_norm_for_2_sites(mps, i, D)
            end
        end
        return mps
    end

    function push_N(arr::Array{Any,1}, op::Array{Complex{Float64},2}, N::Int64)
        arr_ = copy(arr)
        for i in 1:N
            push!(arr_, op)
        end
        return arr_
    end
    
    function j1j2_connection(L::Int64)
        J1_term = Set()
        for i in 1:L
            for j in 1:(L-1)
                row = 1 + 4*(i-1)
                push!(J1_term, (row + j - 1, row + j))
                col = i
                push!(J1_term, (col + 4*(j - 1), col + 4 + 4*(j - 1)))
            end
        end

        J2_term = Set()
        for i in 1:(L-1)
            for j in 1:(L-1)
                row = 1 + 4*(i-1)
                push!(J2_term, (row + (j - 1), row + (j - 1) + 5))
                push!(J2_term, (row + j, row + j + 3))
            end
        end
        return (J1_term, J2_term)
    end

    function j1j2_2D_Hamiltonian_MPO(L::Int64, j1::Float64, j2::Float64)
        N = L^2
        max_distance = L + 1
        MPO_L = 3*(max_distance + 2)
        O = []
        first_MPO = []
        (pauliX, pauliY, pauliZ, pauliI) = pauli()
        zero = pauli_zero()
        ops = [pauliX, pauliY, pauliZ]
        (J1_term, J2_term) = j1j2_connection(L)
        for op in ops
            push!(first_MPO, zero)
            for k in 1:max_distance
                if (1, 1 + k) in J1_term
                    push!(first_MPO, op*j1)
                elseif (1, 1 + k) in J2_term
                    push!(first_MPO, op*j2)
                else
                    push!(first_MPO, zero)
                end
            end
            push!(first_MPO, zero)
        end
        first_MPO_tuple = Tuple(first_MPO)
        push!(O, dstack(first_MPO_tuple))
    
        for site in 1:N-2
            mid_MPO = []
            for i in 1:length(ops)
                for row in 1:(max_distance + 2)
                    tmp_row = []
                    for j in 1:(i-1) # fill zero
                        tmp_row = push_N(tmp_row, zero, max_distance + 2)
                    end
    
                    if row == 1
                        push!(tmp_row, pauliI)
                        tmp_row = push_N(tmp_row, zero, max_distance + 1)
                    elseif row == 2
                        push!(tmp_row, ops[i])
                        tmp_row = push_N(tmp_row, zero, max_distance + 1)
                    elseif row == (max_distance + 2)
                        push!(tmp_row, zero)
                        for k in 1:max_distance
                            if (site, site + k) in J1_term
                                push!(tmp_row, j1*ops[i])
                            elseif (site, site + k) in J1_term
                                push!(tmp_row, j2*ops[i])
                            else
                                push!(tmp_row, zero)
                            end
                        end
                        push!(tmp_row, zero)
                    else
                        tmp_row = push_N(tmp_row, zero, row - 2)
                        push!(tmp_row, pauliI)
                        tmp_row = push_N(tmp_row, zero, max_distance + 1 - (row - 2))
                    end
                    for j in 1:(3-i) # fill zero
                        tmp_row = push_N(tmp_row, zero, max_distance + 2)
                    end
                    tmp_row_tuple = Tuple(tmp_row)
                    push!(mid_MPO, tmp_row_tuple)
                end
            end
            push!(O, ddstack(mid_MPO))
        end
    
        last_MPO = []
        for op in ops
            push!(last_MPO, pauliI)
            push!(last_MPO, op)
            last_MPO = push_N(last_MPO, zero, max_distance)
        end
        last_MPO_tuple = Tuple(last_MPO)
        push!(O, dstack(last_MPO_tuple))
        return O
    end

end