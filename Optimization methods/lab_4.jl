using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Printf

mkpath("lab8_output")

rosen(x)    = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
rastr(x)    = 20 + x[1]^2 + x[2]^2 - 10 * (cos(2π * x[1]) + cos(2π * x[2]))
schwefel(x) = 418.9829 * 2 - (x[1] * sin(sqrt(abs(x[1]))) + x[2] * sin(sqrt(abs(x[2]))))

function get_quadratic_min(x0, x1, x2, y0, y1, y2)
    d01 = x1 - x0;  d02 = x2 - x0;  d12 = x2 - x1
    (abs(d01) < 1e-14 || abs(d02) < 1e-14) && return nothing
    a1 = (y1 - y0) / d01
    a2 = ((y2 - y0) / d02 - (y1 - y0) / d01) / d12
    a2 <= 0 && return nothing
    return (x0 + x1) / 2.0 - a1 / (2.0 * a2)
end

function line_search(f, x_base, dim; delta=0.01, eps=1e-8)
    curr_x = copy(x_base)
    for _ in 1:200
        x0 = copy(curr_x)
        x1 = copy(x0); x1[dim] += delta
        y0, y1 = f(x0), f(x1)
        if y0 > y1
            x2 = copy(x0); x2[dim] += 2.0 * delta
        else
            x2 = copy(x0); x2[dim] -= delta
        end
        y2 = f(x2)
        x_star_val = get_quadratic_min(x0[dim], x1[dim], x2[dim], y0, y1, y2)
        val_m, idx_m = findmin([y0, y1, y2])
        best_triple  = [x0, x1, x2][idx_m]
        if isnothing(x_star_val)
            new_x = best_triple
        else
            x_star = copy(x0); x_star[dim] = x_star_val
            y_star = f(x_star)
            new_x = y_star < val_m ? x_star : best_triple
        end
        norm(new_x - curr_x) < eps && return new_x
        curr_x = new_x
    end
    return curr_x
end

function pauell_method(f, x0; delta_x=0.01, delta_y=0.01, eps=1e-9, maxiter=5000)
    x_curr = copy(x0)
    traj   = [copy(x0)]
    for _ in 1:maxiter
        x_old  = copy(x_curr)
        x_curr = line_search(f, x_curr, 1; delta=delta_x, eps=eps * 0.1)
        x_curr = line_search(f, x_curr, 2; delta=delta_y, eps=eps * 0.1)
        push!(traj, copy(x_curr))
        norm(x_curr - x_old) < eps && break
    end
    return x_curr, [f(x) for x in traj], traj
end

function build_figure(title_str, f, xl, yl, traj; levels=40, n=150, optimum=[0.0, 0.0])
    xs = range(xl[1], xl[2], length=n)
    ys = range(yl[1], yl[2], length=n)
    Z  = [f([xi, yi]) for yi in ys, xi in xs]
    p2 = contour(xs, ys, Z; levels=levels, fill=true, color=:viridis,
                 title="$title_str (2D)", xlabel="x₁", ylabel="x₂")
    p3 = surface(xs, ys, Z; color=:viridis, alpha=0.6,
                 title="$title_str (3D)", colorbar=false)
    tx, ty = [p[1] for p in traj], [p[2] for p in traj]
    tz = [f(p) for p in traj]
    plot!(p2, tx, ty; color=:red, lw=1.2, label="Траектория")
    plot!(p3, tx, ty, tz; color=:red, lw=2, label="")
    scatter!(p2, [tx[1]],      [ty[1]];      markershape=:diamond, markercolor=:white,  label="Старт")
    scatter!(p2, [tx[end]],    [ty[end]];    markershape=:hexagon, markercolor=:red,    label="Финиш")
    scatter!(p2, [optimum[1]], [optimum[2]]; markershape=:star5,   markercolor=:yellow, label="Оптимум")
    fig = plot(p2, p3; layout=(1, 2), size=(1400, 600), margin=5mm)
    display(fig)
    return fig
end

all_results = []

# Розенброк: мелкий шаг для узкой долины, больше итераций
xf, fv, tr = pauell_method(rosen, [-1.2, 1.0]; delta_x=0.005, delta_y=0.005, eps=1e-9, maxiter=5000)
push!(all_results, ("Розенброк", "Пауэлл", length(fv)-1, fv[end], xf))
savefig(build_figure("Розенброк", rosen, (-2, 2), (-1, 3), tr; optimum=[1.0, 1.0]), "lab8_output/rosenbrock.png")

# Растригин: старт [0.1, -0.1] — близко к глобальному минимуму [0,0],
# шаг 0.05 — достаточно мал, чтобы не перескочить в соседний локальный минимум
xf, fv, tr = pauell_method(rastr, [0.1, -0.1]; delta_x=0.05, delta_y=0.05, eps=1e-9, maxiter=5000)
push!(all_results, ("Растригин", "Пауэлл", length(fv)-1, fv[end], xf))
savefig(build_figure("Растригин", rastr, (-3, 3), (-3, 3), tr; optimum=[0.0, 0.0]), "lab8_output/rastrigin.png")

# Швефель: крупный шаг, старт [400, 400]
xf, fv, tr = pauell_method(schwefel, [400.0, 400.0]; delta_x=1.0, delta_y=1.0, eps=1e-9, maxiter=5000)
push!(all_results, ("Швефель", "Пауэлл", length(fv)-1, fv[end], xf))
savefig(build_figure("Швефель", schwefel, (300, 500), (300, 500), tr; optimum=[420.9687, 420.9687]), "lab8_output/schwefel.png")

println("\n" * "=" ^ 105)
@printf("%-22s %-20s %8s %15s %28s\n", "Функция", "Метод", "Итер.", "f*", "x*")
println("-" ^ 105)
for (fname, mname, iters, fval, xval) in all_results
    xstr = @sprintf("[%.6f, %.6f]", xval[1], xval[2])
    @printf("%-22s %-20s %8d %15.4e %28s\n", fname, mname, iters, fval, xstr)
end
println("=" ^ 105)

