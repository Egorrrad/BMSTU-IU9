
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Printf

mkpath("lab9_output")

rosen(x)    = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
rastr(x)    = 20 + x[1]^2 + x[2]^2 - 10 * (cos(2π * x[1]) + cos(2π * x[2]))
schwefel(x) = 418.9829 * 2 - (x[1] * sin(sqrt(abs(x[1]))) + x[2] * sin(sqrt(abs(x[2]))))

function gradient(f, x; h=1e-5)
    n = length(x)
    g = zeros(n)
    for i in 1:n
        xp, xm = copy(x), copy(x)
        xp[i] += h;  xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2h)
    end
    return g
end

function hessian(f, x; h=1e-4)
    n = length(x)
    H = zeros(n, n)
    for i in 1:n, j in 1:n
        xpp, xpm, xmp, xmm = copy(x), copy(x), copy(x), copy(x)
        xpp[i] += h;  xpp[j] += h
        xpm[i] += h;  xpm[j] -= h
        xmp[i] -= h;  xmp[j] += h
        xmm[i] -= h;  xmm[j] -= h
        H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4h^2)
    end
    return H
end

function newton(f, x0; eps=1e-6, maxiter=1000)
    x = copy(x0)
    traj = [copy(x)]
    for _ in 1:maxiter
        g = gradient(f, x)
        norm(g) < eps && break
        H = hessian(f, x)
        d = -H \ g
        alpha = 1.0
        while f(x + alpha * d) >= f(x) && alpha > 1e-10
            alpha *= 0.5
        end
        x = x + alpha * d
        push!(traj, copy(x))
    end
    return x, [f(p) for p in traj], traj
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
    for i in 1:length(traj)-1
        dx = tx[i+1] - tx[i]
        dy = ty[i+1] - ty[i]
        quiver!(p2, [tx[i]], [ty[i]]; quiver=([dx], [dy]),
                color=:orange, lw=1.5, label=(i == 1 ? "Направления" : ""))
    end
    scatter!(p2, [tx[1]],      [ty[1]];      markershape=:diamond, markercolor=:white,  label="Старт")
    scatter!(p2, [tx[end]],    [ty[end]];    markershape=:hexagon, markercolor=:red,    label="Финиш")
    scatter!(p2, [optimum[1]], [optimum[2]]; markershape=:star5,   markercolor=:yellow, label="Оптимум")
    fig = plot(p2, p3; layout=(1, 2), size=(1400, 600), margin=5mm)
    display(fig)
    return fig
end

all_results = []

xf, fv, tr = newton(rosen, [-1.2, 1.0])
push!(all_results, ("Розенброк", "Ньютон", length(fv)-1, fv[end], xf))
savefig(build_figure("Розенброк", rosen, (-2,2), (-1,3), tr; optimum=[1.0, 1.0]), "lab9_output/rosenbrock_newton.png")

xf, fv, tr = newton(rastr, [0.1, -0.1])
push!(all_results, ("Растригин", "Ньютон", length(fv)-1, fv[end], xf))
savefig(build_figure("Растригин", rastr, (-3,3), (-3,3), tr; optimum=[0.0, 0.0]), "lab9_output/rastrigin_newton.png")

xf, fv, tr = newton(schwefel, [400.0, 400.0])
push!(all_results, ("Швефель", "Ньютон", length(fv)-1, fv[end], xf))
savefig(build_figure("Швефель", schwefel, (300,500), (300,500), tr; optimum=[420.9687, 420.9687]), "lab9_output/schwefel_newton.png")

println("
" * "=" ^ 105)
@printf("%-22s %-20s %8s %15s %28s
", "Функция", "Метод", "Итер.", "f*", "x*")
println("-" ^ 105)
for (fname, mname, iters, fval, xval) in all_results
    xstr = @sprintf("[%.6f, %.6f]", xval[1], xval[2])
    @printf("%-22s %-20s %8d %15.4e %28s
", fname, mname, iters, fval, xstr)
end
println("=" ^ 105)

