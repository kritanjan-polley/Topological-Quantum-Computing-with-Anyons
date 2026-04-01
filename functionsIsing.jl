module functionsIsing

using LinearAlgebra
using Printf
using StaticArrays
using CairoMakie

export PROCESS_MEMORY_COUNTERS, ForwardDB
export pic_dir, data_dir
export my_colors, custom_theme
export F, R, sigma1, sigma2, T_gate, generators, I2comp
export gen_syms, sym_mapping, inv_map_sym, inv_idx
export vx, vy, vz
export get_peak_memory_bytes, su2_distance_sq, unitary_to_axis_angle
export axis_angle_to_unitary, GC_decompose, generate_database
export get_path_from_db, search_base_net, invert_path, simplify_path
export solovay_kitaev, decompose_unitary, embed
export to_latex_string

struct PROCESS_MEMORY_COUNTERS
    cb::UInt32
    PageFaultCount::UInt32
    PeakWorkingSetSize::UInt
    WorkingSetSize::UInt
    QuotaPeakPagedPoolUsage::UInt
    QuotaPagedPoolUsage::UInt
    QuotaPeakNonPagedPoolUsage::UInt
    QuotaNonPagedPoolUsage::UInt
    PagefileUsage::UInt
    PeakPagefileUsage::UInt
end

struct ForwardDB
    u::Vector{SMatrix{2,2,ComplexF64,4}}
    parent_idx::Vector{Int}
    gen_idx::Vector{Int}
    keys::Vector{Float64}
    indices::Vector{Int}
end

lw = 1.75
ms = 10
fontsize = 24
ticksize = 20
legendsize = 20
labelsize = 22
legendlabelsize = 22
titlesize = 20
figure_size = (700, 500)
const my_colors = cgrad(:glasbey_bw_minc_20_maxl_70_n256, 256, categorical=true)

const pic_dir = "/Users/kritanjanpolley/Desktop/pic_dir"
const data_dir = joinpath(pwd(), "data_dir")
mkpath(data_dir)

const custom_theme::Attributes = Theme(
    Figure=(size=figure_size,),
    Axis=(
        xgridvisible=true,
        ygridvisible=true,
        xgridstyle=:dash,
        ygridstyle=:dash,
        xticklabelsize=ticksize,
        yticklabelsize=ticksize,
        xlabelsize=labelsize,
        ylabelsize=labelsize,
        xtickformat=x -> [@sprintf("%.1f", val) for val in x],
        markersize=ms,
    ),
    palette=(color=my_colors,),
    Legend=(
        titlesize=titlesize,
        labelsize=legendlabelsize,
        markersize=ms,
        framevisible=false,
        tellwidth=false,
        tellheight=false,
    ),
)

const F = (1.0 / sqrt(2.0)) .* SMatrix{2,2,ComplexF64}(
    1.0, 1.0,
    1.0, -1.0
)

const R = SMatrix{2,2,ComplexF64}(
    exp(-1.0 * im * pi / 8.0), 0.0,
    0.0, exp(3.0 * im * pi / 8.0)
)

const sigma1 = R
const sigma2 = inv(F) * R * F

const T_gate = SMatrix{2,2,ComplexF64}(
    1.0, 0.0,
    0.0, exp(im * pi / 4.0)
)

const generators = SVector{6,SMatrix{2,2,ComplexF64,4}}(
    sigma1, sigma2, T_gate,
    inv(sigma1), inv(sigma2), inv(T_gate)
)
const I2comp = SMatrix{2,2,ComplexF64}(I)

const gen_syms::NTuple{6,Symbol} = (:sigma1, :sigma2, :Tgate, :sigma1i, :sigma2i, :Tgatei)
const sym_mapping = Dict(gen_syms[i] => generators[i] for i in eachindex(generators))

const inv_map_sym = Dict(
    :sigma1 => :sigma1i, :sigma1i => :sigma1,
    :sigma2 => :sigma2i, :sigma2i => :sigma2,
    :Tgate => :Tgatei, :Tgatei => :Tgate
)

const inv_idx = SVector{6,Int}(4, 5, 6, 1, 2, 3)

const vx = SVector{3,Float64}(1, 0, 0)
const vy = SVector{3,Float64}(0, 1, 0)
const vz = SVector{3,Float64}(0, 0, 1)

const latex_replacement_map = Dict(
    :sigma1 => "s_1",
    :sigma2 => "s_2",
    :Tgate => "S_3",
    :sigma1i => "s_1^{-1}",
    :sigma2i => "s_2^{-1}",
    :Tgatei => "s_3^{-1}",
)

function get_peak_memory_bytes()::Int64
    if Sys.islinux() || Sys.isapple()
        rusage = zeros(Int64, 18)
        ret = ccall(:getrusage, Int32, (Int32, Ptr{Cvoid}), 0, rusage)

        if ret == 0
            multiplier = Sys.islinux() ? 1024 : 1
            return rusage[5] * multiplier
        end

    elseif Sys.iswindows()
        hProcess = ccall(:GetCurrentProcess, Ptr{Cvoid}, ())
        mem_counters = Ref(PROCESS_MEMORY_COUNTERS(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        cb = sizeof(PROCESS_MEMORY_COUNTERS)

        ret = ccall((:GetProcessMemoryInfo, "psapi"), Int32,
            (Ptr{Cvoid}, Ptr{PROCESS_MEMORY_COUNTERS}, UInt32),
            hProcess, mem_counters, cb)

        if ret != 0
            return Int(mem_counters[].PeakWorkingSetSize)
        end
    end
    return -1
end

@inline function su2_distance_sq(u, v)
    val = dot(u, v)
    return 1.0 - abs(val) * 0.5
end

function unitary_to_axis_angle(U::SMatrix{2,2,ComplexF64,4})
    phase = det(U)^(-0.5)
    U_norm = U * phase
    tr_val = tr(U_norm)
    cos_t2 = clamp(real(tr_val) * 0.5, -1.0, 1.0)
    theta = 2.0 * acos(cos_t2)
    if abs(theta) < 1e-10
        return 0.0, vz
    end
    sin_t2 = sin(theta * 0.5)
    n = SVector{3,Float64}(-imag(U_norm[1, 2]) / sin_t2,
        -real(U_norm[1, 2]) / sin_t2,
        -imag(U_norm[1, 1]) / sin_t2)
    return theta, normalize(n)
end

function axis_angle_to_unitary(theta::Float64, n::SVector{3,Float64})
    c = cos(theta * 0.5)
    s = sin(theta * 0.5)
    return SMatrix{2,2,ComplexF64}(c - im * s * n[3], -im * s * (n[1] + im * n[2]),
        -im * s * (n[1] - im * n[2]), c + im * s * n[3])
end

function GC_decompose(U::SMatrix{2,2,ComplexF64,4})
    theta, n = unitary_to_axis_angle(U)
    phi_ang = 2.0 * asin(sqrt(sin(theta * 0.25)))
    V_sim = axis_angle_to_unitary(phi_ang, vx)
    W_sim = axis_angle_to_unitary(phi_ang, vy)
    rot_axis = cross(vz, n)
    rot_sin = norm(rot_axis)
    rot_cos = dot(vz, n)
    S = (rot_sin < 1e-10) ? ((rot_cos > 0.0) ? I2comp :
         SMatrix{2,2,ComplexF64}(0, -im, -im, 0)) :
        axis_angle_to_unitary(acos(clamp(rot_cos, -1.0, 1.0)), normalize(rot_axis))
    return S * V_sim * S', S * W_sim * S'
end

function generate_database(depth::Int)
    println("Generating Solovay-Kitaev Instructor set (Length $depth)")
    est_size::Int = 4 * 3^(depth - 1)
    u = [I2comp]
    parent = [0]
    gen = [0]
    sizehint!(u, est_size)
    sizehint!(parent, est_size)
    sizehint!(gen, est_size)

    layer_start = 1
    for unnecessary_idx in 1:depth
        current_end = length(u)
        for p_idx in layer_start:current_end
            last_op = gen[p_idx]
            for i in eachindex(generators)
                if last_op != 0 && i == inv_idx[last_op]
                    continue
                end
                push!(u, generators[i] * u[p_idx])
                push!(parent, p_idx)
                push!(gen, i)
            end
        end
        layer_start = current_end + 1
    end
    keys = real.(getindex.(u, 1))
    p = sortperm(keys)
    return ForwardDB(u, parent, gen, keys[p], Int.(p))
end

function get_path_from_db(db::ForwardDB, idx::Int)
    path = Symbol[]
    curr = idx
    while curr > 1
        push!(path, gen_syms[db.gen_idx[curr]])
        curr = db.parent_idx[curr]
    end
    return path
end

function search_base_net(target_u::SMatrix{2,2,ComplexF64,4}, db::ForwardDB)
    key = real(target_u[1, 1])
    window = 0.05 ## a bit arbitrary
    idx_start = searchsortedfirst(db.keys, key - window)
    idx_end = searchsortedlast(db.keys, key + window)
    idx_start = max(1, idx_start)
    idx_end = min(length(db.indices), idx_end)

    best_dist = 2.0
    best_idx = 0
    if idx_start > idx_end
        idx_start = max(1, searchsortedfirst(db.keys, key) - 50)
        idx_end = min(length(db.indices), idx_start + 100)
    end

    @inbounds for k in idx_start:idx_end
        idx = db.indices[k]
        d = su2_distance_sq(db.u[idx], target_u)
        if d < best_dist
            best_dist = d
            best_idx = idx
        end
    end
    return best_dist, best_idx
end

function invert_path(path::Vector{Symbol})
    return [inv_map_sym[s] for s in reverse(path)]
end

function solovay_kitaev(U::SMatrix{2,2,ComplexF64,4}, depth::Int, db::ForwardDB)
    if depth == 0
        _, idx = search_base_net(U, db)
        return db.u[idx], simplify_path(get_path_from_db(db, idx))
    end
    U_prev, path_prev = solovay_kitaev(U, depth - 1, db)
    Delta = U * U_prev'
    V, W = GC_decompose(Delta)
    _, path_v = solovay_kitaev(V, depth - 1, db)
    _, path_w = solovay_kitaev(W, depth - 1, db)
    path_next = [path_v; path_w; invert_path(path_v); invert_path(path_w); path_prev]
    path_next = simplify_path(path_next)
    res_mat = foldl(*, [sym_mapping[s] for s in path_next]; init=I2comp)
    return res_mat, path_next
end

function decompose_unitary(U_in::AbstractMatrix)
    U = Matrix{ComplexF64}(U_in)
    N = size(U, 1)
    gates = []

    for j in 1:N-1
        for i in j+1:N
            a = U[j, j]
            b = U[i, j]

            if abs(b) < 1e-9
                continue
            end

            if abs(a) < 1e-9
                c = 0.0
                s = conj(b) / abs(b)
            else
                r = hypot(abs(a), abs(b))
                if r == 0.0
                    c = 1.0
                    s = 0.0
                else
                    c = abs(a) / r
                    s = (a / abs(a)) * (conj(b) / r)
                end
            end

            for k in 1:N
                val_j = U[j, k]
                val_i = U[i, k]
                U[j, k] = c * val_j + s * val_i
                U[i, k] = -conj(s) * val_j + c * val_i
            end
            gate_inv = SMatrix{2,2,ComplexF64}(c, conj(s), -s, c)
            push!(gates, (j, i, gate_inv))
        end
    end
    return gates, Diagonal(U)
end

function embed(U2::AbstractMatrix, N::Int, i::Int, j::Int)
    res = Matrix{ComplexF64}(I, N, N)
    res[i, i] = U2[1, 1]
    res[i, j] = U2[1, 2]
    res[j, i] = U2[2, 1]
    res[j, j] = U2[2, 2]
    return res
end

const YB_reductions = Dict(
    (:sigma1, :sigma2, :sigma1, :sigma2i) => (:sigma2, :sigma1),
    (:sigma2i, :sigma1, :sigma2, :sigma1) => (:sigma1, :sigma2),
    (:sigma2, :sigma1, :sigma2, :sigma1i) => (:sigma1, :sigma2),
    (:sigma1i, :sigma2, :sigma1, :sigma2) => (:sigma2, :sigma1),
    (:sigma1i, :sigma2i, :sigma1i, :sigma2) => (:sigma2i, :sigma1i),
    (:sigma2, :sigma1i, :sigma2i, :sigma1i) => (:sigma1i, :sigma2i),
    (:sigma2i, :sigma1i, :sigma2i, :sigma1) => (:sigma1i, :sigma2i),
    (:sigma1, :sigma2i, :sigma1i, :sigma2i) => (:sigma2i, :sigma1i)
)


function simplify_path(path::Vector{Symbol})
    function clean(p)
        s = Symbol[]
        sizehint!(s, length(p))
        for x in p
            if !isempty(s) && s[end] == inv_map_sym[x]
                pop!(s)
            else
                push!(s, x)
                if length(s) >= 4 && (s[end] in (:sigma1, :sigma2, :sigma1i, :sigma2i))
                    if s[end] == s[end-1] == s[end-2] == s[end-3]
                        resize!(s, length(s) - 4)
                        continue
                    end
                end
                if length(s) >= 8 && (s[end] in (:Tgate, :Tgatei))
                    if s[end] == s[end-1] == s[end-2] == s[end-3] == s[end-4] == s[end-5] == s[end-6] == s[end-7]
                        resize!(s, length(s) - 8)
                    end
                end
            end
        end
        return s
    end

    current_path = clean(path)

    while true
        prev_len = length(current_path)
        new_path = Symbol[]
        sizehint!(new_path, prev_len)

        i = 1
        while i <= prev_len
            segment = i <= prev_len - 3 ? (current_path[i], current_path[i+1],
                current_path[i+2], current_path[i+3]) : nothing

            if haskey(YB_reductions, segment)
                push!(new_path, YB_reductions[segment]...)
                i += 4
            else
                push!(new_path, current_path[i])
                i += 1
            end
        end

        current_path = clean(new_path)
        length(current_path) == prev_len && break
    end

    return current_path
end

function to_latex_string(arr::Vector{Symbol})
    result_strings = [latex_replacement_map[s] for s in arr]
    return join(result_strings, " ")
end

end # module end
