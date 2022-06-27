

function construct_tr_matrixes(dicts; upper_dict::Bool=true)
    A_all = []
    T_all = []

    for v in dicts
        all_d = Dict()
        A = []
        T = []

        for (i, lv) in v
            #println(lv)
            d_tmp = Dict()
            for (t,a) in lv
                if haskey(d_tmp, a)
                    if upper_dict
                        d_tmp[a] = maximum([d_tmp[a], t])
                    else
                        d_tmp[a] = minimum([d_tmp[a], t])
                    end
                else
                    d_tmp[a] = t
                end
            end
            all_d[i] = d_tmp
        end

        for (i, ld) in all_d
            for (a, t) in ld
                push!(A, a)
                push!(T, t)
            end
        end
        push!(A_all, A)
        push!(T_all, T)

    end
    return A_all, T_all
end

function find_closest_planes(a, t, A_all, T_all)
    
    a_list = []
    t_list = []
    for i=1:length(A_all)
        A = Matrix(hcat(A_all[i]...)');
        sse = sum((hcat(A_all[i]...)'.-repeat(1.0*(a'), size(A_all[i],1))).^2,dims=2)[:]
        close_idxes = [k for (k,v) in filter((x)->x[2]<0.1, collect(enumerate(sse)))];
        if length(close_idxes) > 0
            best_idx = collect(filter((x)-> x ∈ close_idxes, sortperm(abs.(t.-1.0*T_all[i]))))[1]
            push!(a_list, A_all[i][best_idx])
            push!(t_list, T_all[i][best_idx])
        end
    end
    if length(t_list) == 0
        a_list = [a]
        t_list = [t]
    end
    
    push!(a_list, a)
    push!(t_list, t)
    
    return a_list, t_list
end