
using Plots
using LinearAlgebra
using Printf
using Colors
gr()
mkpath("lab42_output")

function rosenbrock3(x)
    return (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2 +
           (1.0 - x[2])^2 + 100.0*(x[3] - x[2]^2)^2
end

function rastrigin3(x)
    return 30.0 + sum(x[i]^2 - 10.0*cos(2π*x[i]) for i in 1:3)
end

function schwefel3(x)
    return 418.9829*3 - sum(x[i]*sin(sqrt(abs(x[i]))) for i in 1:3)
end

TEST_FUNCTIONS = [
    (f=rosenbrock3, name="Розенброк", x0=[-2.0,-2.0,-2.0],
     x_opt=[1.0,1.0,1.0], f_opt=0.0),
    (f=rastrigin3, name="Растригин", x0=[0.5,0.5,0.5],
     x_opt=[0.0,0.0,0.0], f_opt=0.0),
    (f=schwefel3, name="Швефель", x0=[0.5,0.5,0.5],
     x_opt=[420.9687,420.9687,420.9687], f_opt=0.0),
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
                        eps::Float64=1e-6, alpha::Float64=1.0,
                        max_iter::Int=8000)
    n = length(x0)
    simplex = create_initial_simplex(x0, alpha)
    history = [copy(simplex)]
    trajectory = [copy(x0)]
    iterations = 0; no_imp = 0
    last_best = f(simplex[argmin([f(s) for s in simplex])])
    while iterations < max_iter
        iterations += 1
        fvals = [f(s) for s in simplex]
        idx = sortperm(fvals)
        simplex = simplex[idx]; fvals = fvals[idx]
        f_best = fvals[1]
        centroid = sum(simplex[1:end-1]) / n
        if maximum(norm(simplex[i] - centroid) for i in 1:n+1) < eps; break; end
        if abs(f_best - last_best) < eps
            no_imp += 1; if no_imp > 30; break; end
        else
            no_imp = 0; last_best = f_best
        end
        x_worst = simplex[end]
        x_r = centroid + 1.0 * (centroid - x_worst)
        if f(x_r) < fvals[end]
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
    return simplex[argmin(fvals)], trajectory, history, iterations
end

function nelder_mead(f, x0::Vector{Float64};
                     eps::Float64=1e-6, alpha_nm::Float64=1.0,
                     rho::Float64=1.0, chi::Float64=2.0,
                     gamma::Float64=0.5, sigma::Float64=0.5,
                     max_iter::Int=8000)
    n = length(x0)
    simplex = create_initial_simplex(x0, alpha_nm)
    history = [copy(simplex)]
    trajectory = [copy(x0)]
    iterations = 0; no_imp = 0
    last_best = f(simplex[argmin([f(s) for s in simplex])])
    while iterations < max_iter
        iterations += 1
        fvals = [f(s) for s in simplex]
        idx = sortperm(fvals)
        simplex = simplex[idx]; fvals = fvals[idx]
        f_best = fvals[1]; f_worst = fvals[end]; f_2nd = fvals[end-1]
        centroid = sum(simplex[1:end-1]) / n
        if maximum(norm(simplex[i] - centroid) for i in 1:n+1) < eps; break; end
        if abs(f_best - last_best) < eps
            no_imp += 1; if no_imp > 30; break; end
        else
            no_imp = 0; last_best = f_best
        end
        x_worst = simplex[end]
        x_r = centroid + rho * (centroid - x_worst); f_r = f(x_r)
        if f_r < f_best
            x_e = centroid + chi * (x_r - centroid); f_e = f(x_e)
            simplex[end] = f_e < f_r ? x_e : x_r
            push!(trajectory, copy(simplex[end]))
        elseif f_r < f_2nd
            simplex[end] = x_r; push!(trajectory, copy(x_r))
        elseif f_r < f_worst
            x_c = centroid + gamma * (x_r - centroid); f_c = f(x_c)
            if f_c <= f_r
                simplex[end] = x_c; push!(trajectory, copy(x_c))
            else
                x_best = simplex[1]
                for i in 2:n+1; simplex[i] = x_best + sigma*(simplex[i]-x_best); end
                push!(trajectory, copy(simplex[1]))
            end
        else
            x_c = centroid - gamma * (centroid - x_worst); f_c = f(x_c)
            if f_c < f_worst
                simplex[end] = x_c; push!(trajectory, copy(x_c))
            else
                x_best = simplex[1]
                for i in 2:n+1; simplex[i] = x_best + sigma*(simplex[i]-x_best); end
                push!(trajectory, copy(simplex[1]))
            end
        end
        push!(history, copy(simplex))
    end
    fvals = [f(s) for s in simplex]
    return simplex[argmin(fvals)], trajectory, history, iterations
end

const TETRA_EDGES = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

function adaptive_lims(hist_ss, hist_nm, traj_ss, traj_nm, ax::Int; pad=0.15)
    vals = Float64[]
    for h in [hist_ss, hist_nm]
        for simp in h, v in simp
            push!(vals, v[ax])
        end
    end
    for t in [traj_ss, traj_nm], v in t
        push!(vals, v[ax])
    end
    lo, hi = minimum(vals), maximum(vals)
    margin = max((hi - lo) * pad, 0.5)
    return lo - margin, hi + margin
end

function draw_tetrahedra_3d!(p, history; max_shown=50, lw=2.0, first_label="Тетраэдры")
    total = length(history)
    step = max(1, total ÷ max_shown)
    idxs = unique([1; collect(1:step:total); total])
    m = length(idxs)
    labeled = false
    for (k, i) in enumerate(idxs)
        simp = history[i]
        t = (k - 1) / max(m - 1, 1)
        col = RGB(t, 1.0 - 0.5*t, 0.0)
        for (a, b) in TETRA_EDGES
            lbl = (!labeled) ? first_label : ""
            plot3d!(p,
                    [simp[a][1], simp[b][1]],
                    [simp[a][2], simp[b][2]],
                    [simp[a][3], simp[b][3]],
                    linecolor=col, linewidth=lw, label=lbl)
            labeled = true
        end
    end
end

function draw_tetrahedra_proj!(p, history, ax1::Int, ax2::Int;
                               max_shown=50, lw=2.0, first_label="Тетраэдры")
    total = length(history)
    step = max(1, total ÷ max_shown)
    idxs = unique([1; collect(1:step:total); total])
    m = length(idxs)
    labeled = false
    for (k, i) in enumerate(idxs)
        simp = history[i]
        t = (k - 1) / max(m - 1, 1)
        col = RGB(t, 1.0 - 0.5*t, 0.0)
        for (a, b) in TETRA_EDGES
            lbl = (!labeled) ? first_label : ""
            plot!(p,
                  [simp[a][ax1], simp[b][ax1]],
                  [simp[a][ax2], simp[b][ax2]],
                  color=col, linewidth=lw, label=lbl)
            labeled = true
        end
    end
end

function plot_comparison(tf, res_ss, traj_ss, hist_ss, it_ss,
                         res_nm, traj_nm, hist_nm, it_nm)
    f = tf.f
    fname = tf.name
    x0 = tf.x0
    lx = adaptive_lims(hist_ss, hist_nm, traj_ss, traj_nm, 1)
    ly = adaptive_lims(hist_ss, hist_nm, traj_ss, traj_nm, 2)
    lz = adaptive_lims(hist_ss, hist_nm, traj_ss, traj_nm, 3)

    # Убраны все background_color=:black — теперь стандартный светлый фон
    p1 = plot3d(title="Простой симплекс | $(fname)\niter=$(it_ss) f≈$(round(f(res_ss),sigdigits=5))",
                xlabel="x₁", ylabel="x₂", zlabel="x₃",
                legend=:topright, camera=(30,30),
                xlims=lx, ylims=ly, zlims=lz)
    draw_tetrahedra_3d!(p1, hist_ss; first_label="Тетраэдры ПС")
    if length(traj_ss) > 1
        st = max(1, length(traj_ss) ÷ 400); ts = traj_ss[1:st:end]
        plot3d!(p1, [v[1] for v in ts],[v[2] for v in ts],[v[3] for v in ts],
                linecolor=:cyan, linewidth=2, label="Траектория")
    end
    scatter3d!(p1, [x0[1]],[x0[2]],[x0[3]], color=:yellow, markersize=8, label="Старт")
    scatter3d!(p1, [res_ss[1]],[res_ss[2]],[res_ss[3]], color=:magenta,
               markersize=9, markershape=:star5, label="Финиш")

    p2 = plot3d(title="Нелдер-Мид | $(fname)\niter=$(it_nm) f≈$(round(f(res_nm),sigdigits=5))",
                xlabel="x₁", ylabel="x₂", zlabel="x₃",
                legend=:topright, camera=(30,30),
                xlims=lx, ylims=ly, zlims=lz)
    draw_tetrahedra_3d!(p2, hist_nm; first_label="Тетраэдры НМ")
    if length(traj_nm) > 1
        st = max(1, length(traj_nm) ÷ 400); tn = traj_nm[1:st:end]
        plot3d!(p2, [v[1] for v in tn],[v[2] for v in tn],[v[3] for v in tn],
                linecolor=:lime, linewidth=2, label="Траектория")
    end
    scatter3d!(p2, [x0[1]],[x0[2]],[x0[3]], color=:yellow, markersize=8, label="Старт")
    scatter3d!(p2, [res_nm[1]],[res_nm[2]],[res_nm[3]], color=:magenta,
               markersize=9, markershape=:star5, label="Финиш")

    p3 = plot(title="ПС | Проекция x₁–x₂", xlabel="x₁", ylabel="x₂",
              legend=:topright, xlims=lx, ylims=ly)
    draw_tetrahedra_proj!(p3, hist_ss, 1, 2; first_label="Тетраэдры ПС")
    if length(traj_ss) > 1
        st = max(1, length(traj_ss) ÷ 300); ts = traj_ss[1:st:end]
        plot!(p3, [v[1] for v in ts],[v[2] for v in ts],
              color=:cyan, linewidth=2, label="Траектория")
    end
    scatter!(p3, [x0[1]],[x0[2]], color=:yellow, markersize=8, label="Старт")
    scatter!(p3, [res_ss[1]],[res_ss[2]], color=:magenta, markersize=9,
             markershape=:star5, label="Финиш")

    p4 = plot(title="НМ | Проекция x₁–x₂", xlabel="x₁", ylabel="x₂",
              legend=:topright, xlims=lx, ylims=ly)
    draw_tetrahedra_proj!(p4, hist_nm, 1, 2; first_label="Тетраэдры НМ")
    if length(traj_nm) > 1
        st = max(1, length(traj_nm) ÷ 300); tn = traj_nm[1:st:end]
        plot!(p4, [v[1] for v in tn],[v[2] for v in tn],
              color=:lime, linewidth=2, label="Траектория")
    end
    scatter!(p4, [x0[1]],[x0[2]], color=:yellow, markersize=8, label="Старт")
    scatter!(p4, [res_nm[1]],[res_nm[2]], color=:magenta, markersize=9,
             markershape=:star5, label="Финиш")

    best_ss = [f(hist_ss[i][argmin([f(s) for s in hist_ss[i]])]) for i in 1:length(hist_ss)]
    best_nm = [f(hist_nm[i][argmin([f(s) for s in hist_nm[i]])]) for i in 1:length(hist_nm)]
    best_ss_pos = max.(best_ss .- tf.f_opt .+ 1e-12, 1e-12)
    best_nm_pos = max.(best_nm .- tf.f_opt .+ 1e-12, 1e-12)

    p5 = plot(1:length(best_ss), log10.(best_ss_pos),
              color=:cyan, linewidth=2.5, label="Простой симплекс",
              xlabel="Итерация", ylabel="log₁₀(f − f*)",
              title="Конвергенция | $(fname)", legend=:topright)
    plot!(p5, 1:length(best_nm), log10.(best_nm_pos),
          color=:orange, linewidth=2.5, label="Нелдер-Мид")

    p6 = plot(title="Проекция x₂–x₃", xlabel="x₂", ylabel="x₃",
              legend=:topright, xlims=ly, ylims=lz)
    draw_tetrahedra_proj!(p6, hist_ss, 2, 3; first_label="Тетраэдры ПС")
    draw_tetrahedra_proj!(p6, hist_nm, 2, 3; max_shown=40, lw=1.5, first_label="Тетраэдры НМ")
    if length(traj_ss) > 1
        st = max(1, length(traj_ss) ÷ 300); ts = traj_ss[1:st:end]
        plot!(p6, [v[2] for v in ts],[v[3] for v in ts],
              color=:cyan, linewidth=2, label="Траект. ПС")
    end
    if length(traj_nm) > 1
        st = max(1, length(traj_nm) ÷ 300); tn = traj_nm[1:st:end]
        plot!(p6, [v[2] for v in tn],[v[3] for v in tn],
              color=:orange, linewidth=2, label="Траект. НМ")
    end
    scatter!(p6, [x0[2]],[x0[3]], color=:yellow, markersize=8, label="Старт")
    scatter!(p6, [res_ss[2]],[res_ss[3]], color=:magenta, markersize=8,
             markershape=:star5, label="Финиш ПС")
    scatter!(p6, [res_nm[2]],[res_nm[3]], color=:cyan, markersize=8,
             markershape=:star5, label="Финиш НМ")

    return plot(p1, p2, p3, p4, p5, p6,
                layout=(2,3), size=(1700,1000),
                plot_title="Сравнение: Простой симплекс vs Нелдер-Мид | $fname (3D)",
                plot_titlefontsize=13)
end

function convergence_plot(results)
    names = [r.fname for r in results]
    iters_ss = [r.iters_ss for r in results]
    iters_nm = [r.iters_nm for r in results]
    errs_ss = [max(r.err_ss, 1e-12) for r in results]
    errs_nm = [max(r.err_nm, 1e-12) for r in results]
    x = 1:length(names); w = 0.35
    p1 = bar(x .- w/2, iters_ss, bar_width=w, color=:steelblue,
             label="Простой симплекс", xticks=(x, names),
             ylabel="Итерации", title="Количество итераций",
             legend=:topright, xrotation=10)
    bar!(p1, x .+ w/2, iters_nm, bar_width=w, color=:coral, label="Нелдер-Мид")
    p2 = bar(x .- w/2, log10.(errs_ss), bar_width=w, color=:steelblue,
             label="Простой симплекс", xticks=(x, names),
             ylabel="log₁₀|f − f*|", title="Точность (лог. масштаб)",
             legend=:topright, xrotation=10)
    bar!(p2, x .+ w/2, log10.(errs_nm), bar_width=w, color=:coral, label="Нелдер-Мид")
    return plot(p1, p2, layout=(1,2), size=(1000,450),
                plot_title="Сравнительный анализ методов (3D функции)",
                plot_titlefontsize=12)
end

println("="^75)
println(" ЛАБ. 4.2 — Сравнение: Простой Симплекс vs Нелдер-Мид (3D функции)")
println("="^75)
println()
@printf("%-14s | %-22s | %-22s\n", "Функция", "Простой симплекс", "Нелдер-Мид")
@printf("%-14s | %-8s %-13s | %-8s %-13s\n", "", "Итер.", "f(x*)", "Итер.", "f(x*)")
println("-"^65)
summary = []
for tf in TEST_FUNCTIONS
    f = tf.f; fname = tf.name; x0 = tf.x0
    res_ss, traj_ss, hist_ss, it_ss = simple_simplex(f, copy(x0))
    res_nm, traj_nm, hist_nm, it_nm = nelder_mead(f, copy(x0))
    fss = f(res_ss); fnm = f(res_nm)
    err_ss = abs(fss - tf.f_opt); err_nm = abs(fnm - tf.f_opt)
    @printf("%-14s | %-8d %-13.6g | %-8d %-13.6g\n", fname, it_ss, fss, it_nm, fnm)
    push!(summary, (fname=fname, iters_ss=it_ss, iters_nm=it_nm,
                    f_ss=fss, f_nm=fnm, err_ss=err_ss, err_nm=err_nm,
                    res_ss=res_ss, res_nm=res_nm))
    plt = plot_comparison(tf, res_ss, traj_ss, hist_ss, it_ss,
                          res_nm, traj_nm, hist_nm, it_nm)
    display(plt)
    fn = replace(fname, " "=>"_")
    savefig(plt, "lab42_comparison_$(fn).png")
    println(" → lab42_comparison_$(fn).png")
end
println("-"^65)
println()
println()
println("="^75)
println(" ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ")
println("="^75)
for r in summary
    println()
    println("Функция: $(r.fname)")
    @printf(" Простой симплекс → x* = [%8.4f, %8.4f, %8.4f] f = %-12.6g iter = %d\n",
            r.res_ss[1], r.res_ss[2], r.res_ss[3], r.f_ss, r.iters_ss)
    @printf(" Нелдер-Мид → x* = [%8.4f, %8.4f, %8.4f] f = %-12.6g iter = %d\n",
            r.res_nm[1], r.res_nm[2], r.res_nm[3], r.f_nm, r.iters_nm)
    @printf(" Соотношение итераций ПС/НМ: %.2f\n",
            r.iters_nm > 0 ? r.iters_ss / r.iters_nm : NaN)
end

