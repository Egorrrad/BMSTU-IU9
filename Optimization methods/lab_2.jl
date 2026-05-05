
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Printf

mkpath("lab5_output")

quad(x, α, β) = α * x[1]^2 + β * x[2]^2
∇quad(x, α, β) = [2α * x[1], 2β * x[2]]

rastr(x) = 20 + x[1]^2 + x[2]^2 - 10 * (cos(2π * x[1]) + cos(2π * x[2]))
∇rastr(x) = [2x[1] + 20π * sin(2π * x[1]), 2x[2] + 20π * sin(2π * x[2])]

function golden_section(phi, a, b; tol=1e-12, maxiter=200)
    tau = (√5 - 1) / 2
    x1 = b - tau * (b - a)
    x2 = a + tau * (b - a)
    f1 = phi(x1)
    f2 = phi(x2)
    for _ in 1:maxiter
        (b - a) < tol && break
        if f1 < f2
            b = x2; x2 = x1; f2 = f1
            x1 = b - tau * (b - a); f1 = phi(x1)
        else
            a = x1; x1 = x2; f1 = f2
            x2 = a + tau * (b - a); f2 = phi(x2)
        end
    end
    return (a + b) / 2
end

function gradient_descent(f, grad_f, x0; alpha_0=1.0, eps=1e-6, maxiter=1000)
    x = copy(x0)
    traj = [copy(x)]; fv = [f(x)]
    for _ in 1:maxiter
        g = grad_f(x)
        norm(g) <= eps && break
        y = -g
        alpha = alpha_0
        fx = f(x)
        for _ in 1:50
            f(x + alpha * y) < fx && break
            alpha /= 2
        end
        x = x + alpha * y
        push!(traj, copy(x)); push!(fv, f(x))
    end
    return x, fv, traj
end

function steepest_descent(f, grad_f, x0; eps=1e-6, maxiter=1000, alpha_max=10.0)
    x = copy(x0)
    traj = [copy(x)]; fv = [f(x)]
    for _ in 1:maxiter
        g = grad_f(x)
        norm(g) <= eps && break
        y = -g
        phi(alpha) = f(x + alpha * y)
        alpha_opt = golden_section(phi, 0.0, alpha_max)
        x = x + alpha_opt * y
        push!(traj, copy(x)); push!(fv, f(x))
    end
    return x, fv, traj
end

function conjugate_gradients(f, grad_f, x0, beta_formula::Symbol, A_diag; eps=1e-6, maxiter=1000, alpha_max=10.0, N_restart=0)
    n = length(x0)
    if N_restart == 0
        N_restart = n
    end
    x = copy(x0)
    traj = [copy(x)]; fv = [f(x)]
    g_prev = grad_f(x)
    d = -copy(g_prev)
    for k in 1:maxiter
        g_cur = grad_f(x)
        norm(g_cur) <= eps && break
        if k > 1
            if mod(k - 1, N_restart) == 0
                beta = 0.0
            else
                delta_g = g_cur - g_prev
                if beta_formula == :fletcher_reeves
                    beta = dot(g_cur, g_cur) / dot(g_prev, g_prev)
                elseif beta_formula == :polak_ribiere
                    beta = dot(g_cur, delta_g) / dot(g_prev, g_prev)
                elseif beta_formula == :hestenes_stiefel
                    denom = dot(d, delta_g)
                    beta = abs(denom) < 1e-30 ? 0.0 : -dot(g_cur, delta_g) / denom
                elseif beta_formula == :dixon
                    denom = dot(d, g_prev)
                    beta = abs(denom) < 1e-30 ? 0.0 : -dot(g_cur, g_cur) / denom
                elseif beta_formula == :dai_yuan
                    denom = dot(d, delta_g)
                    beta = abs(denom) < 1e-30 ? 0.0 : dot(g_cur, delta_g) / denom
                else
                    error("Неизвестная формула beta: $beta_formula")
                end
            end
            d = -g_cur + beta * d
        end
        g_prev = copy(g_cur)
        if !iszero(A_diag)
            Ad = A_diag .* d
            dAd = dot(d, Ad)
            abs(dAd) < 1e-30 && break
            alpha_opt = -dot(g_cur, d) / dAd
        else
            phi(alpha) = f(x + alpha * d)
            alpha_opt = golden_section(phi, 0.0, alpha_max)
        end
        x = x + alpha_opt * d
        push!(traj, copy(x)); push!(fv, f(x))
    end
    return x, fv, traj
end

const COLORS = [:gray, :black, :red, :blue, :green, :orange, :purple]
const ALL_METHODS = [(:gd, "Градиентный спуск", :gray), (:sd, "Наискорейший спуск", :black), (:fletcher_reeves, "Флетчер-Ривз", :red), (:polak_ribiere, "Полак-Рибьер", :blue), (:hestenes_stiefel, "Хестенс-Стифель", :green), (:dixon, "Диксон", :orange), (:dai_yuan, "Дайян", :purple)]

function run_methods(f, grad_f, x0, A_diag; enabled=Symbol[], alpha_0=0.01, eps=1e-6, alpha_max=50.0, maxiter=1000)
    methods_to_run = isempty(enabled) ? [m[1] for m in ALL_METHODS] : enabled
    results = []
    for (id, name, color) in ALL_METHODS
        id in methods_to_run || continue
        if id == :gd
            xf, fv, tr = gradient_descent(f, grad_f, x0; alpha_0=alpha_0, eps=eps, maxiter=maxiter)
        elseif id == :sd
            xf, fv, tr = steepest_descent(f, grad_f, x0; eps=eps, maxiter=maxiter, alpha_max=alpha_max)
        else
            xf, fv, tr = conjugate_gradients(f, grad_f, x0, id, A_diag; eps=eps, maxiter=maxiter, alpha_max=alpha_max)
        end
        push!(results, (id=id, name=name, color=color, x=xf, fv=fv, traj=tr))
    end
    return results
end

function make_grid(f; xlims=(-5,5), ylims=(-5,5), n=150)
    xs = range(xlims[1], xlims[2], n)
    ys = range(ylims[1], ylims[2], n)
    zs = [f([xi, yi]) for yi in ys, xi in xs]
    return xs, ys, zs
end

function build_figure(title_str, f, xl, yl, results; levels=30, n=150, optimum=[0.0, 0.0])
    xs, ys, Z = make_grid(f; xlims=xl, ylims=yl, n=n)
    p3 = surface(xs, ys, Z; color=:viridis, alpha=0.7, xlabel="x₁", ylabel="x₂", zlabel="f", title="$title_str (3D)", colorbar=false)
    p2 = contour(xs, ys, Z; levels=levels, fill=true, color=:viridis, xlabel="x₁", ylabel="x₂", title="$title_str (2D)", colorbar=true, linewidth=0.3)

    for r in results
        plot!(p2, [0], [0]; color=r.color, lw=1.5, label=r.name)
    end

    for r in results
        tx = [p[1] for p in r.traj]
        ty = [p[2] for p in r.traj]
        plot!(p2, tx, ty; color=r.color, linewidth=1.5, label="")
        plot!(p3, tx, ty, [f(p) for p in r.traj]; color=r.color, linewidth=2, label="")
    end

    x0 = results[1].traj[1]
    scatter!(p2, [x0[1]], [x0[2]]; markershape=:diamond, markersize=6, markercolor=:white, markerstrokecolor=:black, markerstrokewidth=2, label="Старт")
    scatter!(p3, [x0[1]], [x0[2]], [f(x0)]; markershape=:diamond, markersize=5, markercolor=:white, markerstrokecolor=:black, markerstrokewidth=2, label="")

    for r in results
        scatter!(p2, [r.x[1]], [r.x[2]]; markershape=:hexagon, markersize=4, markercolor=r.color, markerstrokecolor=:black, markerstrokewidth=1, label="")
    end

    scatter!(p2, [optimum[1]], [optimum[2]]; markershape=:star5, markersize=6, markercolor=:yellow, markerstrokecolor=:black, markerstrokewidth=1.5, label="Оптимум")

    fig = plot(p2, p3; layout=(1, 2), size=(1600, 650))
    display(fig)
    fig
end

function print_table(all_results)
    println("\n" * "=" ^ 95)
    @printf("%-22s %-20s %8s %15s %22s\n", "Функция", "Метод", "Итер.", "f*", "x*")
    println("-" ^ 95)
    for (fname, method, iters, fval, xval) in all_results
        xstr = @sprintf("[%.6f, %.6f]", xval[1], xval[2])
        @printf("%-22s %-20s %8d %15.8e %22s\n", fname, method, iters, fval, xstr)
    end
    println("=" ^ 95)
end

x0 = [100.0, 100.0]
quad_params = [(1,1), (1,10), (1,20)]
enabled = [:gd, :sd, :fletcher_reeves, :polak_ribiere, :hestenes_stiefel, :dixon, :dai_yuan]
all_results = []

for (a, b) in quad_params
    f(x) = quad(x, a, b)
    grad_f(x) = ∇quad(x, a, b)
    A_diag = [2a, 2b]
    fname = "f = $(a)x² + $(b)x²"
    results = run_methods(f, grad_f, x0, A_diag; enabled=enabled, alpha_0=0.01, eps=1e-6, alpha_max=50.0, maxiter=1000)
    for r in results
        push!(all_results, (fname, r.name, length(r.fv)-1, last(r.fv), r.x))
    end
    lim = 110.0
    fig = build_figure(fname, f, (-lim, lim), (-lim, lim), results)
    savefig(fig, "lab5_output/quadratic_$(a)_$(b).png")
end
print_table(all_results)

x0_rast = [4.0, 4.0]
f_rast(x) = rastr(x)
grad_rast(x) = ∇rastr(x)
A_diag_dummy = [0.0, 0.0]
results_rast = run_methods(f_rast, grad_rast, x0_rast, A_diag_dummy; enabled=enabled, alpha_0=0.01, eps=1e-6, alpha_max=1.0, maxiter=1000)
fig_rast = build_figure("Растригин", f_rast, (-5,5), (-5,5), results_rast; levels=30)
savefig(fig_rast, "lab5_output/rastrigin.png")

println("\n--- Растригин ---")
@printf("%-20s %8s %15s %22s\n", "Метод", "Итер.", "f*", "x*")
println("-" ^ 70)
for r in results_rast
    xstr = @sprintf("[%.6f, %.6f]", r.x[1], r.x[2])
    @printf("%-20s %8d %15.8e %22s\n", r.name, length(r.fv)-1, last(r.fv), xstr)
end

println("\n✓ Все файлы сохранены в lab5_output/")

