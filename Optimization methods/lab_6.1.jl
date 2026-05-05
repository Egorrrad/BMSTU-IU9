
using LinearAlgebra
using Plots

if !isdir("lab13")
    mkdir("lab13")
end

# Метод золотого сечения
function golden_section_search(f, a, b; tol=1e-8, max_iter=10000)
    if a > b
        a, b = b, a
    end
    gr = (√5 - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    iter = 0
    while (b - a) > tol && iter < max_iter
        if fc <= fd
            b = d; d = c; fd = fc
            c = b - gr * (b - a); fc = f(c)
        else
            a = c; c = d; fc = fd
            d = a + gr * (b - a); fd = f(d)
        end
        iter += 1
    end
    x = (a + b) / 2
    return x, f(x), iter
end

H(h) = h^2
G(g) = g > 0 ? g^2 : 0.0

function Q(x, r::Vector{Float64}, f, h_list::Vector{Function}, g_list::Vector{Function})
    m = length(h_list)
    penalty = f(x)
    for j in 1:m
        penalty += r[j] * H(h_list[j](x))
    end
    for j in 1:length(g_list)
        penalty += r[m+j] * G(g_list[j](x))
    end
    return penalty
end

function violation(x, h_list, g_list)
    return sum(H(h(x)) for h in h_list) + sum(G(g(x)) for g in g_list)
end

# Метод внешних штрафов
function exterior_penalty_method_1d(f, h_list, g_list, x0, a, b;
    r0=1.0, beta=10.0, tol_x=1e-10, tol_viol=1e-12, max_outer=50, inner_tol=1e-12)

    k = length(h_list) + length(g_list)
    r = fill(r0, k)
    x = clamp(x0, min(a,b), max(a,b))

    for _ in 1:max_outer
        Qf = x -> Q(x, r, f, h_list, g_list)
        x_new, _, _ = golden_section_search(Qf, a, b; tol=inner_tol)
        x_new = clamp(x_new, min(a,b), max(a,b))
        if abs(x_new - x) <= tol_x && violation(x_new, h_list, g_list) <= tol_viol
            return x_new
        end
        x = x_new
        r .*= beta
    end
    return x
end

# Линейный поиск
function backtracking_step(f, x, fx, g; α0=1.0, c=1e-4, ρ=0.5, max_ls=50)
    α = α0
    gg = dot(g, g)
    for _ in 1:max_ls
        xn = x .- α .* g
        fn = f(xn)
        if fn <= fx - c * α * gg
            return xn, fn, α
        end
        α *= ρ
    end
    xn = x .- α .* g
    return xn, f(xn), α
end

# Градиентный спуск
function gradient_descent(f, gradf, x0; tol_g=1e-10, tol_x=1e-12, max_iter=50000)
    x = copy(x0)
    fx = f(x)
    α = 1.0
    for _ in 1:max_iter
        g = gradf(x)
        if norm(g) <= tol_g
            return x
        end
        xn, fn, α_new = backtracking_step(f, x, fx, g; α0=α)
        if norm(xn - x) <= tol_x
            return xn
        end
        x = xn; fx = fn
        α = min(1.0, α_new)
    end
    return x
end

# Метод внешних штрафов
function exterior_penalty_method_2d(f, gradf, g_list::Vector{Function}, gradg_list::Vector{Function}, x0;
    r0=1.0, beta=10.0, tol_x=1e-10, tol_violation=1e-12, max_outer=20,
    inner_tol_g=1e-10, inner_tol_x=1e-12, max_inner=50_000)

    if length(g_list) != length(gradg_list)
        error("Длины g_list и gradg_list не совпадают: $(length(g_list)) vs $(length(gradg_list))")
    end

    r = fill(r0, length(g_list))
    x_prev = copy(x0)
    outer = 0

    while outer < max_outer
        function Qpen(x)
            s = f(x)
            for (j, gj) in enumerate(g_list)
                v = gj(x)
                if v > 0
                    s += r[j] * v^2
                end
            end
            return s
        end

        function gradQpen(x)
            g = gradf(x)
            for (j, gj) in enumerate(g_list)
                v = gj(x)
                if v > 0
                    g .+= (2 * r[j] * v) .* gradg_list[j](x)
                end
            end
            return g
        end

        # Минимизация штрафной функции
        x_new = gradient_descent(Qpen, gradQpen, x_prev;
            tol_g=inner_tol_g, tol_x=inner_tol_x, max_iter=max_inner)

        # Невязка ограничений
        v = sum(max(0.0, gj(x_new))^2 for gj in g_list)

        if norm(x_new - x_prev) <= tol_x && v <= tol_violation
            x_prev = x_new
            break
        end

        x_prev = x_new
        r .*= beta
        outer += 1
    end

    return x_prev, f(x_prev), outer, r
end

function ellipse_shape(cx, cy, rx, ry; n=220)
    θ = range(0, 2π, length=n)
    return cx .+ rx .* cos.(θ), cy .+ ry .* sin.(θ)
end

# Визуализация 1D
function visualize_1d_example()
    f(x) = x^2
    g(x) = 2 - x
    Qr(x, r) = f(x) + r * (g(x) > 0 ? g(x)^2 : 0.0)

    xs = range(-0.5, 3.5, length=1000)
    r_list = [1.0, 10.0, 100.0, 10000.0, 100000.0, 1000000.0]

    plt = plot(xs, Qr.(xs, r_list[1]), label="r = $(r_list[1])",
        title="Exterior penalty method (1D)",
        xlabel="x", ylabel="Q(x, r)", lw=2.5, palette=:tab10,
        size=(950, 550), legend=:topright)

    for r in r_list[2:end]
        plot!(xs, Qr.(xs, r), label="r = $r", lw=2.5)
    end

    vspan!([2.0, 3.5], label="допустимая область x >= 2", color=:green, alpha=0.12)
    vline!([2.0], label="граница x=2", lc=:red, ls=:dash, lw=3)

    for r in r_list
        Qf = x -> Qr(x, r)
        x_star, _, _ = golden_section_search(Qf, -0.5, 3.5; tol=1e-12)
        scatter!([x_star], [Qr(x_star, r)], label="", mc=:blue, ms=7)
    end

    display(plt)
    savefig(plt, "lab13/lab13_1d_all_r.png")
    println("Сохранён: lab13/lab13_1d_all_r.png")
end

# Визуализация 2D
function visualize_2d_example()
    f(x) = x[1]^2 + 2*x[2]^2
    gradf(x) = [2*x[1], 4*x[2]]

    g_list     = [x -> 2 - x[1], x -> 2 - x[2]]
    gradg_list = [x -> [-1.0, 0.0], x -> [0.0, -1.0]]

    r_list = [1.0, 10.0, 100.0, 10000.0, 100000.0, 1000000.0]
    x0 = [1.2, 1.0]

    xs_path = [copy(x0)]
    x = copy(x0)

    println("2D результаты:")
    for r in r_list
        x, _, _, _ = exterior_penalty_method_2d(f, gradf, g_list, gradg_list, x; r0=r)
        push!(xs_path, copy(x))
        println("   r = $r → x ≈ [$(round(x[1], digits=5)), $(round(x[2], digits=5))]")
    end

    x1s = range(0, 3.5, length=220)
    x2s = range(0, 3.5, length=220)
    ex, ey = ellipse_shape(2.0, 2.0, 0.45, 0.45)
    oval = Shape(ex, ey)

    for (i, r) in enumerate(r_list)
        Z = [begin
            s = f([x1, x2])
            for gj in g_list
                v = gj([x1, x2])
                s += r * (v > 0 ? v^2 : 0.0)
            end
            s
        end for x2 in x2s, x1 in x1s]

        plt = contour(x1s, x2s, Z, levels=40, fill=true, c=:viridis,
            title="При r = $r", xlabel="x1", ylabel="x2", size=(850,550))

        plot!(oval, label="допустимая область", color=:green, alpha=0.18)
        vline!([2.0], lc=:white, ls=:dash, lw=2)
        hline!([2.0], lc=:white, ls=:dash, lw=2)

        pathx = [p[1] for p in xs_path[1:i+1]]
        pathy = [p[2] for p in xs_path[1:i+1]]
        plot!(pathx, pathy, lc=:red, lw=3, label="траектория")
        scatter!(pathx, pathy, mc=:red, ms=6, label=false)

        display(plt)
        savefig(plt, "lab13/lab13_2d_contour_r$(round(Int,r)).png")
    end

    Zf = [f([x1,x2]) for x2 in x2s, x1 in x1s]
    traj = contour(x1s, x2s, Zf, levels=30, fill=false, c=:grays,
        title="Траектория метода внешних штрафов (на f(x))",
        xlabel="x1", ylabel="x2", size=(850,550))

    plot!(traj, oval, label="допустимая область", color=:green, alpha=0.18)
    vline!([2.0], lc=:black, ls=:dash, lw=2)
    hline!([2.0], lc=:black, ls=:dash, lw=2)

    pathx = [p[1] for p in xs_path]
    pathy = [p[2] for p in xs_path]
    plot!(traj, pathx, pathy, lc=:red, lw=4, label="полная траектория")
    scatter!(traj, pathx, pathy, mc=:red, ms=7, label="точки по r")

    display(traj)
    savefig(traj, "lab13/lab13_2d_trajectory.png")
    println("Сохранён: lab13/lab13_2d_trajectory.png")
end

visualize_1d_example()
visualize_2d_example()

