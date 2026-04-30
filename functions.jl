module functions

using LinearAlgebra
using Printf
using StaticArrays
using CairoMakie
using SparseArrays
using NearestNeighbors

export PROCESS_MEMORY_COUNTERS, ForwardDB
export pic_dir, data_dir
export my_colors, custom_theme
export phi, F, R, sigma1, sigma2, generators, I2comp
export i2, pauliZ, pauliX, numOp
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
    tree::KDTree{SVector{8,Float64},Euclidean,Float64}
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

const pic_dir = joinpath(pwd(), "pic_dir")
const data_dir = joinpath(pwd(), "data_dir")
mkpath(data_dir)
mkpath(pic_dir)

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

const phi::Float64 = (1.0 + sqrt(5.0)) * 0.5
const F = SMatrix{2,2,ComplexF64}(1.0 / phi, 1.0 / sqrt(phi), 1.0 / sqrt(phi), -1.0 / phi)
const R = SMatrix{2,2,ComplexF64}(exp(-4.0 * im * pi / 5.0), 0.0, 0.0, exp(-2.0 * im * pi / 5.0))

const sigma1 = R
const sigma2 = inv(F) * R * F
const generators = SVector{4,SMatrix{2,2,ComplexF64,4}}(sigma1, sigma2, inv(sigma1), inv(sigma2))
const I2comp = SMatrix{2,2,ComplexF64}(I)

const gen_syms::NTuple{4,Symbol} = (:sigma1, :sigma2, :sigma1i, :sigma2i)
const sym_mapping = Dict(gen_syms[i] => generators[i] for i in 1:4)
const inv_map_sym = Dict(:sigma1 => :sigma1i, :sigma1i => :sigma1, :sigma2 => :sigma2i, :sigma2i => :sigma2)
const inv_idx = SVector{4,Int}(3, 4, 1, 2)

const vx = SVector{3,Float64}(1, 0, 0)
const vy = SVector{3,Float64}(0, 1, 0)
const vz = SVector{3,Float64}(0, 0, 1)

const i2::Matrix{ComplexF64} = [1.0 0.0; 0.0 1.0]
const pauliX::Matrix{ComplexF64} = [0.0 1.0; 1.0 0.0]
const pauliZ::Matrix{ComplexF64} = [1.0 0.0; 0.0 -1.0]
const numOp::Matrix{ComplexF64} = [0.0 0.0; 0.0 1.0]

latex_replacement_map = Dict(
    :sigma1 => "s_1",
    :sigma2 => "s_2",
    :sigma1i => "s_1^{-1}",
    :sigma2i => "s_2^{-1}"
)

function get_peak_memory_bytes()::Int64
    if Sys.islinux() || Sys.isapple()
        rusage = zeros(Int64, 18)
        ret = ccall(:getrusage, Int32, (Int32, Ptr{Cvoid}), 0, rusage)

        if ret == 0
            # rusage[1-2] = utime, rusage[3-4] = stime, rusage[5] = maxrss
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

@inline function to_r8(U::SMatrix{2,2,ComplexF64,4})
    return SVector{8,Float64}(
        real(U[1, 1]), imag(U[1, 1]),
        real(U[1, 2]), imag(U[1, 2]),
        real(U[2, 1]), imag(U[2, 1]),
        real(U[2, 2]), imag(U[2, 2])
    )
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
    return SMatrix{2,2,ComplexF64}(c - im * s * n[3],
        -im * s * (n[1] + im * n[2]),
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
            for i in 1:4
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
    r4_points = [to_r8(mat) for mat in u]
    tree = KDTree(r4_points)
    return ForwardDB(u, parent, gen, tree)
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
    v = to_r8(target_u)
    idx_plus, dist_plus = knn(db.tree, v, 1)
    idx_minus, dist_minus = knn(db.tree, -v, 1)

    if dist_plus[1] < dist_minus[1]
        best_idx = idx_plus[1]
    else
        best_idx = idx_minus[1]
    end

    best_dist = su2_distance_sq(db.u[best_idx], target_u)

    return best_dist, best_idx
end

function invert_path(path::Vector{Symbol})
    return [inv_map_sym[s] for s in reverse(path)]
end

function solovay_kitaev(U::SMatrix{2,2,ComplexF64,4},
    depth::Int, db::ForwardDB; tol::Float64=1e-10)
    if depth == 0
        _, idx = search_base_net(U, db)
        return db.u[idx], get_path_from_db(db, idx)
    end
    U_prev, path_prev = solovay_kitaev(U, depth - 1, db; tol=tol)

    if su2_distance_sq(U, U_prev) < tol
        # println("Stopping early")
        return U_prev, path_prev
    end

    Delta = U * U_prev'
    V, W = GC_decompose(Delta)
    # _, path_v = solovay_kitaev(V, depth - 1, db; tol=tol)
    # _, path_w = solovay_kitaev(W, depth - 1, db; tol=tol)
    # path_next = [path_v; path_w; invert_path(path_v); invert_path(path_w); path_prev]
    # res_mat = foldl(*, [sym_mapping[s] for s in path_next]; init=I2comp)
    V_approx, path_v = solovay_kitaev(V, depth - 1, db; tol=tol)
    W_approx, path_w = solovay_kitaev(W, depth - 1, db; tol=tol)
    path_next = [path_v; path_w; invert_path(path_v); invert_path(path_w); path_prev]
    res_mat = V_approx * W_approx * V_approx' * W_approx' * U_prev
    return res_mat, path_next
end


# function decompose_unitary2(U_in::AbstractMatrix)
#     U = U_in isa SparseMatrixCSC ? copy(U_in) : Matrix{ComplexF64}(U_in)
#     N = size(U, 1)
#     gates = Tuple{Int,Int,SMatrix{2,2,ComplexF64,4}}[]

#     sizehint!(gates, div(N * (N - 1), 2))

#     @inbounds for j in 1:N-1
#         for i in j+1:N
#             a = U[j, j]
#             b = U[i, j]

#             if abs(b) < 1e-9
#                 continue
#             end

#             if abs(a) < 1e-9
#                 c = 0.0
#                 s = conj(b) / abs(b)
#             else
#                 r = hypot(a, b)
#                 c = abs(a) / r
#                 s = (a / abs(a)) * (conj(b) / r)
#             end

#             for k in j:N
#                 val_j = U[j, k]
#                 val_i = U[i, k]

#                 if abs(val_j) > 1e-12 || abs(val_i) > 1e-12
#                     U[j, k] = c * val_j + s * val_i
#                     U[i, k] = -conj(s) * val_j + c * val_i
#                 end
#             end

#             gate_inv = SMatrix{2,2,ComplexF64}(c, conj(s), -s, c)
#             push!(gates, (j, i, gate_inv))
#         end
#     end
#     return gates, Diagonal(U)
# end

function decompose_unitary(U_in::AbstractMatrix)
    U = U_in isa SparseMatrixCSC ? copy(U_in) : Array(U_in)
    N = size(U, 1)

    gates = Tuple{Int,Int,SMatrix{2,2,ComplexF64,4}}[]
    sizehint!(gates, div(N * (N - 1), 2))

    @inbounds for j in 1:N-1
        for i in j+1:N
            if abs(U[i, j]) < 1e-9
                continue
            end

            G, _ = givens(U[j, j], U[i, j], j, i)
            lmul!(G, view(U, :, j:N))

            c, s = G.c, G.s
            gate_inv = SMatrix{2,2,ComplexF64}(c, conj(s), -s, c)
            push!(gates, (j, i, gate_inv))
        end
    end

    return gates, Diagonal(U)
end


function embed(U2::AbstractMatrix, N::Int, i::Int, j::Int; if_sparse=false)
    if if_sparse
        res = sparse(ComplexF64, I, N, N)
    else
        res = Matrix{ComplexF64}(I, N, N)
    end
    res[i, i] = U2[1, 1]
    res[i, j] = U2[1, 2]
    res[j, i] = U2[2, 1]
    res[j, j] = U2[2, 2]
    return res
end

function simplify_path(path::Vector{Symbol})
    # Yang-Baxter reductions
    reductions = Dict(
        (:sigma1, :sigma2, :sigma1, :sigma2i) => (:sigma2, :sigma1),
        (:sigma2i, :sigma1, :sigma2, :sigma1) => (:sigma1, :sigma2),
        (:sigma2, :sigma1, :sigma2, :sigma1i) => (:sigma1, :sigma2),
        (:sigma1i, :sigma2, :sigma1, :sigma2) => (:sigma2, :sigma1),
        (:sigma1i, :sigma2i, :sigma1i, :sigma2) => (:sigma2i, :sigma1i),
        (:sigma2, :sigma1i, :sigma2i, :sigma1i) => (:sigma1i, :sigma2i),
        (:sigma2i, :sigma1i, :sigma2i, :sigma1) => (:sigma1i, :sigma2i),
        (:sigma1, :sigma2i, :sigma1i, :sigma2i) => (:sigma2i, :sigma1i)
    )

    function clean(p)
        s = Symbol[]
        sizehint!(s, length(p))
        for x in p
            if !isempty(s) && s[end] == inv_map_sym[x]
                pop!(s)
            else
                push!(s, x)
                ## s^5 = I
                if length(s) >= 5
                    if s[end] == s[end-1] == s[end-2] == s[end-3] == s[end-4]
                        resize!(s, length(s) - 5)
                    end
                end
                if length(s) >= 4
                    if s[end] == s[end-1] == s[end-2] == s[end-3]
                        val = s[end]
                        resize!(s, length(s) - 4)
                        push!(s, inv_map_sym[val])
                    end
                end
                if length(s) >= 3
                    if s[end] == s[end-1] == s[end-2]
                        val = s[end]
                        inv_val = inv_map_sym[val]
                        resize!(s, length(s) - 3)
                        push!(s, inv_val)
                        push!(s, inv_val)
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

            if haskey(reductions, segment)
                push!(new_path, reductions[segment]...)
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
