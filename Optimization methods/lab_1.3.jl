
using Plots
using LinearAlgebra
using Printf
gr()

function rosenbrock(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function rastrigin(x)
    n = length(x)
    return 20.0 + sum(x[i]^2 - 10.0 * cos(2π * x[i]) for i in 1:n)
end

function schwefel(x)
    n = length(x)
    return 418.9829 * n - sum(x[i] * sin(sqrt(abs(x[i]))) for i in 1:n)
end

TEST_FUNCTIONS = [
    (f=rosenbrock, name="Розенброк",  x0=[-2.0, -2.0], xlim=(-2.5,2.5),   ylim=(-2.5,2.5),   x_opt=[1.0,1.0],           f_opt=0.0),
    (f=rastrigin,  name="Растригин",  x0=[0.5,  0.5],  xlim=(-5.5,5.5),   ylim=(-5.5,5.5),   x_opt=[0.0,0.0],           f_opt=0.0),
    (f=schwefel,   name="Швефель",    x0=[0.5,  0.5],  xlim=(-10.0,10.0), ylim=(-10.0,10.0), x_opt=[420.9687,420.9687],  f_opt=0.0),
]

function create_initial_simplex(x0::Vector{Float64}, alpha::Float64=1.0)
    n = length(x0)
    simplex = [copy(x0)]
    q = (sqrt(n + 1.0) - 1.0) / (n * sqrt(2.0))
    p = q + 1.0 / sqrt(2.0)
    for i in 1:n
        xi = copy(x0)
        for j in 1:n
            xi[j] += alpha * (j == i ? p : q)
        end
        push!(simplex, xi)
    end
    return simplex
end


function simple_simplex(f, x0::Vector{Float64};
                        eps::Float64=1e-6,
                        alpha::Float64=1.0,
                        max_iter::Int=5000)
    n = length(x0)
    simplex  = create_initial_simplex(x0, alpha)
    history  = [copy(simplex)]
    trajectory = [copy(x0)]
    iterations = 0
    no_imp = 0
    last_best = f(simplex[argmin([f(s) for s in simplex])])

    while iterations < max_iter
        iterations += 1
        fvals = [f(s) for s in simplex]
        idx   = sortperm(fvals)
        simplex = simplex[idx]
        fvals   = fvals[idx]

        f_best   = fvals[1]
        centroid = sum(simplex[1:end-1]) / n

        if maximum(norm(simplex[i] - centroid) for i in 1:n) < eps; break; end
        if abs(f_best - last_best) < eps
            no_imp += 1
            if no_imp > 20; break; end
        else
            no_imp = 0
            last_best = f_best
        end

        x_worst = simplex[end]
        x_r = centroid + 1.0 * (centroid - x_worst)
        f_r = f(x_r)

        if f_r < fvals[end]
            simplex[end] = x_r
            push!(trajectory, copy(x_r))
        else
            x_best = simplex[1]
            for i in 2:n+1
                simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
            end
            push!(trajectory, copy(simplex[1]))
        end
        push!(history, copy(simplex))
    end

    fvals = [f(s) for s in simplex]
    best  = simplex[argmin(fvals)]
    return best, trajectory, history, iterations
end


function nelder_mead(f, x0::Vector{Float64};
                     eps::Float64=1e-6,
                     alpha_nm::Float64=1.0,
                     rho::Float64=1.0,
                     chi::Float64=2.0,
                     gamma::Float64=0.5,
                     sigma::Float64=0.5,
                     max_iter::Int=5000)
    n = length(x0)
    simplex  = create_initial_simplex(x0, alpha_nm)
    history  = [copy(simplex)]
    trajectory = [copy(x0)]
    iterations = 0
    no_imp = 0
    last_best = f(simplex[argmin([f(s) for s in simplex])])

    while iterations < max_iter
        iterations += 1
        fvals = [f(s) for s in simplex]
        idx   = sortperm(fvals)
        simplex = simplex[idx]
        fvals   = fvals[idx]

        f_best  = fvals[1]
        f_worst = fvals[end]
        f_2nd   = fvals[end-1]
        centroid = sum(simplex[1:end-1]) / n

        if maximum(norm(simplex[i] - centroid) for i in 1:n) < eps; break; end
        if abs(f_best - last_best) < eps
            no_imp += 1
            if no_imp > 20; break; end
        else
            no_imp = 0
            last_best = f_best
        end

        x_worst = simplex[end]

        # 1. Отражение
        x_r = centroid + rho * (centroid - x_worst)
        f_r = f(x_r)

        if f_r < f_best
            # 2. Растяжение
            x_e = centroid + chi * (x_r - centroid)
            f_e = f(x_e)
            simplex[end] = f_e < f_r ? x_e : x_r
            push!(trajectory, copy(simplex[end]))
        elseif f_r < f_2nd
            simplex[end] = x_r
            push!(trajectory, copy(x_r))
        else
            if f_r < f_worst
                # 3a. Внешнее сжатие
                x_c = centroid + gamma * (x_r - centroid)
                f_c = f(x_c)
                if f_c <= f_r
                    simplex[end] = x_c
                    push!(trajectory, copy(x_c))
                else
                    x_best = simplex[1]
                    for i in 2:n+1; simplex[i] = x_best + sigma*(simplex[i]-x_best); end
                    push!(trajectory, copy(simplex[1]))
                end
            else
                # 3b. Внутреннее сжатие
                x_c = centroid - gamma * (centroid - x_worst)
                f_c = f(x_c)
                if f_c < f_worst
                    simplex[end] = x_c
                    push!(trajectory, copy(x_c))
                else
                    x_best = simplex[1]
                    for i in 2:n+1; simplex[i] = x_best + sigma*(simplex[i]-x_best); end
                    push!(trajectory, copy(simplex[1]))
                end
            end
        end
        push!(history, copy(simplex))
    end

    fvals = [f(s) for s in simplex]
    best  = simplex[argmin(fvals)]
    return best, trajectory, history, iterations
end


function draw_simplexes_2d!(p, history; max_shown=80)
    total = length(history)
    step  = max(1, total ÷ max_shown)
    idxs  = unique([1; collect(1:step:total); total])
    n     = length(idxs)

    for (k, i) in enumerate(idxs)
        simp = history[i]
        pts  = [simp[1], simp[2], simp[3], simp[1]]
        t   = (k - 1) / max(n - 1, 1)
        col = RGB(t, 1.0 - 0.5*t, 0.0)
        plot!(p, [v[1] for v in pts], [v[2] for v in pts],
              color=col, linewidth=1.8, alpha=0.85,
              label=(k == 1 ? "Симплексы" : ""))
    end
end

function draw_simplexes_3d!(p, history, f; max_shown=60)
    total = length(history)
    step  = max(1, total ÷ max_shown)
    idxs  = unique([1; collect(1:step:total); total])
    n     = length(idxs)

    for (k, i) in enumerate(idxs)
        simp = history[i]
        pts  = [simp[1], simp[2], simp[3], simp[1]]
        t   = (k - 1) / max(n - 1, 1)
        col = RGB(t, 1.0 - 0.5*t, 0.0)
        plot3d!(p, [v[1] for v in pts], [v[2] for v in pts], [f(v) for v in pts],
                linecolor=col, linewidth=1.8, alpha=0.85,
                label=(k == 1 ? "Симплексы" : ""))
    end
end


function plot_comparison(tf, res_ss, traj_ss, hist_ss, it_ss,
                             res_nm, traj_nm, hist_nm, it_nm)
    f     = tf.f
    fname = tf.name
    x0    = tf.x0
    xr    = range(tf.xlim[1], tf.xlim[2], length=200)
    yr    = range(tf.ylim[1], tf.ylim[2], length=200)
    Z     = [f([x, y]) for y in yr, x in xr]
    x_opt = tf.x_opt
    in_b(v) = tf.xlim[1] ≤ v[1] ≤ tf.xlim[2] && tf.ylim[1] ≤ v[2] ≤ tf.ylim[2]

    
    p1 = contour(xr, yr, Z, fill=true, color=:thermal, levels=40,
                 xlabel="x₁", ylabel="x₂",
                 title="Простой симплекс | $(fname)\niter=$(it_ss)   f≈$(round(f(res_ss),sigdigits=5))",
                 legend=:topleft, size=(620,520))
    draw_simplexes_2d!(p1, hist_ss)
    if length(traj_ss) > 1
        st = max(1, length(traj_ss) ÷ 300); ts = traj_ss[1:st:end]
        plot!(p1, [v[1] for v in ts], [v[2] for v in ts],
              color=:white, linewidth=2, label="Траектория")
    end
    scatter!(p1, [x0[1]], [x0[2]], color=:yellow, markersize=9, label="Старт")
    scatter!(p1, [res_ss[1]], [res_ss[2]], color=:magenta, markersize=10,
             markershape=:star5, label="Финиш")
    in_b(x_opt) && scatter!(p1, [x_opt[1]], [x_opt[2]], color=:red,
                             markersize=10, markershape=:star8, label="Оптимум")

    
    p2 = contour(xr, yr, Z, fill=true, color=:thermal, levels=40,
                 xlabel="x₁", ylabel="x₂",
                 title="Нелдер-Мид | $(fname)\niter=$(it_nm)   f≈$(round(f(res_nm),sigdigits=5))",
                 legend=:topleft, size=(620,520))
    draw_simplexes_2d!(p2, hist_nm)
    if length(traj_nm) > 1
        st = max(1, length(traj_nm) ÷ 300); tn = traj_nm[1:st:end]
        plot!(p2, [v[1] for v in tn], [v[2] for v in tn],
              color=:white, linewidth=2, label="Траектория")
    end
    scatter!(p2, [x0[1]], [x0[2]], color=:yellow, markersize=9, label="Старт")
    scatter!(p2, [res_nm[1]], [res_nm[2]], color=:magenta, markersize=10,
             markershape=:star5, label="Финиш")
    in_b(x_opt) && scatter!(p2, [x_opt[1]], [x_opt[2]], color=:red,
                             markersize=10, markershape=:star8, label="Оптимум")

    
    p3 = surface(xr, yr, Z, color=:thermal, alpha=0.5,
                 xlabel="x₁", ylabel="x₂", zlabel="f",
                 title="Простой симплекс 3D | $(fname)",
                 camera=(35,55), legend=true, size=(620,520))
    draw_simplexes_3d!(p3, hist_ss, f)
    if length(traj_ss) > 1
        st = max(1, length(traj_ss) ÷ 200); ts = traj_ss[1:st:end]
        plot3d!(p3, [v[1] for v in ts], [v[2] for v in ts], [f(v) for v in ts],
                linecolor=:white, linewidth=2, label="Траектория")
    end
    scatter3d!(p3, [x0[1]], [x0[2]], [f(x0)], color=:yellow, markersize=7, label="Старт")
    scatter3d!(p3, [res_ss[1]], [res_ss[2]], [f(res_ss)], color=:magenta, markersize=8, label="Финиш")

    
    p4 = surface(xr, yr, Z, color=:thermal, alpha=0.5,
                 xlabel="x₁", ylabel="x₂", zlabel="f",
                 title="Нелдер-Мид 3D | $(fname)",
                 camera=(35,55), legend=true, size=(620,520))
    draw_simplexes_3d!(p4, hist_nm, f)
    if length(traj_nm) > 1
        st = max(1, length(traj_nm) ÷ 200); tn = traj_nm[1:st:end]
        plot3d!(p4, [v[1] for v in tn], [v[2] for v in tn], [f(v) for v in tn],
                linecolor=:white, linewidth=2, label="Траектория")
    end
    scatter3d!(p4, [x0[1]], [x0[2]], [f(x0)], color=:yellow, markersize=7, label="Старт")
    scatter3d!(p4, [res_nm[1]], [res_nm[2]], [f(res_nm)], color=:magenta, markersize=8, label="Финиш")

    plot(p1, p2, p3, p4, layout=(2,2), size=(1300,1100),
         plot_title="Сравнение: Простой симплекс vs Нелдер-Мид  |  $fname",
         plot_titlefontsize=13)
end



println("="^70)
println("  СРАВНЕНИЕ: Простой Симплекс vs Метод Нелдера-Мида")
println("="^70)
println()
@printf("%-14s | %-22s | %-22s\n", "Функция", "Простой симплекс", "Нелдер-Мид")
@printf("%-14s | %-8s %-13s | %-8s %-13s\n", "", "Итер.", "f(x*)", "Итер.", "f(x*)")
println("-"^65)

summary = []

for tf in TEST_FUNCTIONS
    f     = tf.f
    fname = tf.name
    x0    = tf.x0

    res_ss, traj_ss, hist_ss, it_ss = simple_simplex(f, copy(x0))
    res_nm, traj_nm, hist_nm, it_nm = nelder_mead(f,    copy(x0))

    fss    = f(res_ss);  fnm    = f(res_nm)
    err_ss = abs(fss - tf.f_opt);  err_nm = abs(fnm - tf.f_opt)

    @printf("%-14s | %-8d %-13.6g | %-8d %-13.6g\n", fname, it_ss, fss, it_nm, fnm)
    push!(summary, (fname=fname, iters_ss=it_ss, iters_nm=it_nm,
                    f_ss=fss, f_nm=fnm, err_ss=err_ss, err_nm=err_nm,
                    res_ss=res_ss, res_nm=res_nm))

    plt = plot_comparison(tf, res_ss, traj_ss, hist_ss, it_ss,
                              res_nm, traj_nm, hist_nm, it_nm)
    display(plt)
    fn = replace(fname, " "=>"_")
    savefig(plt, "comparison_$(fn).png")
    println("  → comparison_$(fn).png")
end

println("-"^65)
println()


println()
println("="^70)
println("  ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ")
println("="^70)
for r in summary
    println()
    println("Функция: $(r.fname)")
    @printf("  Простой симплекс → x* = [%9.5f, %9.5f]   f = %-12.6g   iter = %d\n",
            r.res_ss[1], r.res_ss[2], r.f_ss, r.iters_ss)
    @printf("  Нелдер-Мид       → x* = [%9.5f, %9.5f]   f = %-12.6g   iter = %d\n",
            r.res_nm[1], r.res_nm[2], r.f_nm, r.iters_nm)
    @printf("  Соотношение итераций ПС/НМ: %.2f\n",
            r.iters_nm > 0 ? r.iters_ss / r.iters_nm : NaN)
end

