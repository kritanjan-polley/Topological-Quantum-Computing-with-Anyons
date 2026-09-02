module LSTH

using Downloads
using Libdl
using Printf

export build_lsth, lsth, lsth_collinear, validate_lsth

const source_url = "https://www.cita.utoronto.ca/~boothroy/data/bkmp2/lsth2.f"
const work_dir = joinpath(@__DIR__, "lsth_build")
const src_dir = joinpath(work_dir, "lsth2.f")

function libname()
    if Sys.iswindows()
        return joinpath(work_dir, "liblsth2.dll")
    elseif Sys.isapple()
        return joinpath(work_dir, "liblsth2.dylib")
    else
        return joinpath(work_dir, "liblsth2.so")
    end
end

const _handle = Ref{Ptr{Cvoid}}(C_NULL)
const _symbol = Ref{Ptr{Cvoid}}(C_NULL)

function build_lsth(; force::Bool=false)
    mkpath(work_dir)
    lib = libname()

    if force || !isfile(src_dir)
        @info "Downloading LSTH source" source_url
        Downloads.download(source_url, src_dir)
    end

    if force || !isfile(lib)
        gcc_image = get(ENV, "GCC_IMAGE", "")

        if !isempty(gcc_image)
            isfile(gcc_image) || error(
                "GCC_IMAGE points to a file that does not exist: $gcc_image"
            )

            @info "Compiling LSTH using Apptainer" gcc_image

            if Sys.isapple()
                error("Apptainer compilation is intended for Linux/HPC systems")
            else
                run(`/share/apps/apptainer/bin/apptainer exec $gcc_image gfortran -O3 -fPIC -shared $src_dir -o $lib`)
            end

        else
            gfortran = Sys.which("gfortran")

            if gfortran === nothing && isfile("/usr/bin/gfortran")
                gfortran = "/usr/bin/gfortran"
            end

            gfortran === nothing && error(
                "gfortran was not found and GCC_IMAGE is not set"
            )

            @info "Compiling LSTH using local gfortran" gfortran

            if Sys.isapple()
                run(`$gfortran -O3 -fPIC -dynamiclib $src_dir -o $lib`)
            else
                run(`$gfortran -O3 -fPIC -shared $src_dir -o $lib`)
            end
        end
    end

    if _handle[] != C_NULL
        Libdl.dlclose(_handle[])
    end

    _handle[] = Libdl.dlopen(lib)

    sym = Libdl.dlsym_e(_handle[], :vlsth_)
    sym == C_NULL && error(
        "Could not find Fortran symbol vlsth_ in $lib"
    )

    _symbol[] = sym

    return lib
end

function ensure_loaded()
    if _symbol[] == C_NULL
        build_lsth()
    end
end

function lsth(r12::Real, r23::Real, r13::Real; derivatives::Bool=false)
    ensure_loaded()

    r = Float64[r12, r23, r13]

    any(x -> x <= 0.0, r) &&
        throw(ArgumentError("LSTH requires positive interatomic distances"))

    E  = Ref{Float64}(0.0)
    E1 = Ref{Float64}(0.0)
    E2 = Ref{Float64}(0.0)
    E3 = Ref{Float64}(0.0)

    ideriv = Ref{Int32}(derivatives ? 1 : 0)
    ipr    = Ref{Int32}(0)
    isurf  = Ref{Int32}(0)

    ccall(
        _symbol[],
        Cvoid,
        (
            Ptr{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64},
            Ref{Int32}, Ref{Int32}, Ref{Int32}
        ),
        r, E, E1, E2, E3, ideriv, ipr, isurf
    )

    if derivatives
        return E[], (E1[], E2[], E3[])
    else
        return E[]
    end
end

function lsth_collinear(
    x::Real,
    y::Real;
    derivatives::Bool=false
)
    if !derivatives
        return lsth(x, y, x + y)
    end

    V, d = lsth(x, y, x + y; derivatives=true)

    # Vcol(x,y) = V(x,y,x+y)
    dVdx = d[1] + d[3]
    dVdy = d[2] + d[3]

    return V, (dVdx, dVdy)
end

function validate_lsth()
    build_lsth()

    println("LSTH validation")

    # H2 + H asymptotic geometry
    rH2 = 1.401
    Vasym = lsth(rH2, 20.0, 20.0)

    # Symmetric collinear transition-state region
    rs = 1.757
    Vsad, dsad = lsth(rs, rs, 2rs; derivatives=true)

    barrier = (Vsad - Vasym) * 627.509474

    @printf("H2 + H energy       = %.12f Eh\n", Vasym)
    @printf("symmetric TS energy = %.12f Eh\n", Vsad)
    @printf("classical barrier   = %.6f kcal/mol\n", barrier)

    @printf(
        "TS derivatives      = (% .6e, % .6e, % .6e) Eh/bohr\n",
        dsad...
    )

    # Permutation symmetry test
    V1 = lsth(1.4, 2.0, 2.7)
    V2 = lsth(2.0, 2.7, 1.4)
    V3 = lsth(2.7, 1.4, 2.0)

    @printf(
        "permutation error   = %.3e Eh\n",
        maximum(abs.([V1-V2, V1-V3]))
    )

    # Collinear wrapper check
    Vcol = lsth_collinear(rs, rs)

    @printf(
        "collinear V         = %.12f Eh\n",
        Vcol
    )

    return (
        h2_asymptote = Vasym,
        saddle_energy = Vsad,
        barrier_kcalmol = barrier,
        saddle_derivatives = dsad,
        permutation_error = maximum(abs.([V1-V2, V1-V3]))
    )
end

end # module


if abspath(PROGRAM_FILE) == @__FILE__
    using .LSTH

    LSTH.validate_lsth()

    println("\nExample: collinear H + H2")

    x, y = 1.7570, 1.7570

    V = LSTH.lsth_collinear(x, y)

    println("V($x,$y) = $V Hartree")
end
