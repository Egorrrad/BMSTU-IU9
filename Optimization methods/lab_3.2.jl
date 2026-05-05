

using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Printf
using Colors

mkpath("let4")
# --- 1. Целевые функции ---
rosenbrock(x) = sum(100 * (x[i+1] - x[i]^2)^2 + (x[i] - 1)^2 for i in 1:length(x)-1)
schwefel(x) = 418.9829 * length(x) - sum(xi -> xi * sin(sqrt(abs(xi))), x)
rastrigin(x) = 10 * length(x) + sum(xi -> xi^2 - 10 * cos(2π * xi), x)
# --- 2. Аналитические градиенты ---
function grad_rosenbrock(x)
    n = length(x)
    g = zeros(n)
    for i in 1:n
        if i == 1
            g[i] = -400 * x[1] * (x[2] - x[1]^2) + 2 * (x[1] - 1)
        elseif i == n
            g[i] = 200 * (x[n] - x[n-1]^2) + 2 * (x[n] - 1)
        else
            g[i] = 200 * (x[i] - x[i-1]^2) - 400 * x[i] * (x[i+1] - x[i]^2) + 2 * (x[i] - 1)
        end
    end
    return g
end
function grad_schwefel(x)
    g = similar(x)
    for i in eachindex(x)
        xi = x[i]
        s = sqrt(abs(xi)) + 1e-20
        g[i] = xi >= 0 ? -sin(s) - xi * cos(s) / (2s) : -sin(s) + xi * cos(s) / (2s)
    end
    return g
end
grad_rastrigin(x) = [2 * xi + 20 * sin(2π * xi) for xi in x]
# --- 3. Метод золотого сечения (используется только для одномерного случая λ₂=0) ---
function golden_section(phi, a, b; tol=1e-12, maxiter=200)
    tau = (√5 - 1) / 2
    x1 = b - tau * (b - a)
    x2 = a + tau * (b - a)
    f1 = phi(x1); f2 = phi(x2)
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
# --- 4. Численный градиент (конечные разности) ---
function numerical_gradient(f, x; h=1e-6)
    n = length(x)
    g = zeros(n)
    for i in 1:n
        e = zeros(n); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2h)
    end
    return g
end
# --- 5. Минимизация по λ = [λ₁, λ₂] методом Полака-Рибьера ---
function minimize_polak_ribiere(g, lambda0; tol=1e-8, maxiter=50, alpha_max=2.0)
    lambda = copy(lambda0)
    grad = numerical_gradient(g, lambda)
    d = -copy(grad)
    g_prev = copy(grad)
    for k in 1:maxiter
        norm(grad) < tol && break
        # Одномерный поиск вдоль направления d (золотое сечение)
        phi(alpha) = g(lambda + alpha * d)
        alpha_opt = golden_section(phi, 0.0, alpha_max)
        lambda = lambda + alpha_opt * d
        grad = numerical_gradient(g, lambda)
        # Формула Полака-Рибьера: β = (∇g_new · (∇g_new - ∇g_old)) / ‖∇g_old‖²
        delta_g = grad - g_prev
        denom = dot(g_prev, g_prev)
        beta = denom < 1e-30 ? 0.0 : dot(grad, delta_g) / denom
        beta = max(beta, 0.0) # β⁺ — рестарт при отрицательном β
        # Рестарт каждые n+1 итераций (n=2 для λ)
        k % 3 == 0 && (beta = 0.0)
        d = -grad + beta * d
        g_prev = copy(grad)
    end
    return lambda
end
# --- 6. Метод Миле-Кантрелла ---
function mille_contrell(f, grad_f, x0, eps; max_iter=10000, zero_lambda1=false)
    n = length(x0)
    x = copy(x0)
    x_prev = copy(x0)
    delta_x = zeros(n)
    traj = [copy(x)]
    fv = [f(x)]
    iter_count = 0
    for k in 1:max_iter
        iter_count = k
        if k > 0 && k % (n + 1) == 0
            delta_x = zeros(n)
        end
        g = grad_f(x)
        norm(g) < eps && break
        zero_lambda1 && (delta_x = zeros(n))
        # Вспомогательная функция g(λ) = f(x - λ₁·∇f + λ₂·Δx)
        obj(lambda) = f(x - lambda[1] * g + lambda[2] * delta_x)
        if zero_lambda1
            # Одномерный случай: только λ₁, λ₂=0
            r = max(0.1, min(1.0, norm(delta_x) / (norm(g) + 1e-10)))
            l1_opt = golden_section(t -> obj([t, 0.0]), 0.0, r * 2)
            l_opt = [l1_opt, 0.0]
        else
            # Двумерная оптимизация по [λ₁, λ₂] методом Полака-Рибьера
            r = max(0.1, min(1.0, norm(delta_x) / (norm(g) + 1e-10)))
            lambda0 = [r, 0.0] # начальное приближение
            l_opt = minimize_polak_ribiere(obj, lambda0)
            # Гарантируем λ₁ ≥ 0 (шаг по антиградиенту не может быть отрицательным)
            l_opt[1] = max(l_opt[1], 0.0)
        end
        x_new = x - l_opt[1] * g + l_opt[2] * delta_x
        delta_x = x_new - x
        x_prev = x
        x = x_new
        push!(traj, copy(x))
        push!(fv, f(x))
        norm(delta_x) < eps && break
    end
    return x, fv, traj, iter_count
end
# --- 7. Сетка для графика ---
function make_grid(f; xlims=(-5, 5), ylims=(-5, 5), n=150)
    xs = range(xlims[1], xlims[2], n)
    ys = range(ylims[1], ylims[2], n)
    zs = [f([xi, yi]) for yi in ys, xi in xs]
    return xs, ys, zs
end
# --- 8. Построение фигуры: 2D контур + 3D поверхность ---
function build_figure(title_str, f, xl, yl, traj_mc, traj_gd, x_opt_mc, x_opt_gd, optimum)
    # Подгоняем область под реальные траектории + истинный оптимум
    all_px = vcat([p[1] for p in traj_mc], [p[1] for p in traj_gd], [optimum[1]])
    all_py = vcat([p[2] for p in traj_mc], [p[2] for p in traj_gd], [optimum[2]])
    span_x = max(maximum(all_px) - minimum(all_px), 1.0)
    span_y = max(maximum(all_py) - minimum(all_py), 1.0)
    pad_x = 0.15 * span_x
    pad_y = 0.15 * span_y
    xl = (min(minimum(all_px) - pad_x, xl[1]), max(maximum(all_px) + pad_x, xl[2]))
    yl = (min(minimum(all_py) - pad_y, yl[1]), max(maximum(all_py) + pad_y, yl[2]))
    xs, ys, Z = make_grid(f; xlims=xl, ylims=yl, n=150)
    p3 = surface(xs, ys, Z;
                 color=:viridis, alpha=0.75,
                 xlabel="x₁", ylabel="x₂", zlabel="f",
                 title="$title_str (3D)", colorbar=false)
    p2 = contour(xs, ys, Z;
                 levels=30, fill=true,
                 color=:viridis, colorbar=true, linewidth=0.3,
                 xlabel="x₁", ylabel="x₂",
                 title="$title_str (2D)", legend=:topright)
    # Легенда
    plot!(p2, [0], [0]; color=:white, lw=2, label="МК (Полак-Рибьер)")
    plot!(p2, [0], [0]; color=:tomato, lw=2, label="МК (λ₂=0)")
    # Траектории на 2D
    tx_mc = [p[1] for p in traj_mc]; ty_mc = [p[2] for p in traj_mc]
    tx_gd = [p[1] for p in traj_gd]; ty_gd = [p[2] for p in traj_gd]
    plot!(p2, tx_mc, ty_mc; color=:white, linewidth=2.5, label="")
    plot!(p2, tx_gd, ty_gd; color=:tomato, linewidth=2.5, label="")
    # Траектории на 3D
    plot!(p3, tx_mc, ty_mc, [f(p) for p in traj_mc]; color=:white, linewidth=2, label="")
    plot!(p3, tx_gd, ty_gd, [f(p) for p in traj_gd]; color=:tomato, linewidth=2, label="")
    # Старт
    x0 = traj_mc[1]
    scatter!(p2, [x0[1]], [x0[2]];
             markershape=:diamond, markersize=6,
             markercolor=:white, markerstrokecolor=:black, markerstrokewidth=2,
             label="Старт")
    scatter!(p3, [x0[1]], [x0[2]], [f(x0)];
             markershape=:diamond, markersize=5,
             markercolor=:white, markerstrokecolor=:black, markerstrokewidth=2, label="")
    # Найденные минимумы
    scatter!(p2, [x_opt_mc[1]], [x_opt_mc[2]];
             markershape=:hexagon, markersize=6,
             markercolor=:white, markerstrokecolor=:black, markerstrokewidth=1, label="")
    scatter!(p2, [x_opt_gd[1]], [x_opt_gd[2]];
             markershape=:hexagon, markersize=6,
             markercolor=:tomato, markerstrokecolor=:black, markerstrokewidth=1, label="")
    # Истинный оптимум
    scatter!(p2, [optimum[1]], [optimum[2]];
             markershape=:star5, markersize=8,
             markercolor=:yellow, markerstrokecolor=:black, markerstrokewidth=1.5,
             label="Оптимум")
    fig = plot(p2, p3; layout=(1, 2), size=(1600, 650), margin=5mm)
    display(fig)
    return fig
end
# --- 9. Таблица результатов ---
function print_table(all_results)
    println("\n" * "=" ^ 90)
    @printf("%-14s %-26s %8s %15s %22s\n", "Функция", "Метод", "Итер.", "f*", "x*")
    println("-" ^ 90)
    for (fname, method, iters, fval, xval) in all_results
        xstr = @sprintf("[%.6f, %.6f]", xval[1], xval[2])
        @printf("%-14s %-26s %8d %15.8e %22s\n", fname, method, iters, fval, xstr)
    end
    println("=" ^ 90)
end

function minimize_1d(f, a, b, tol=1e-8)
    for _ in 1:100
        if b - a < tol
            return (a + b) / 2
        end
        c = (2a + b) / 3
        d = (a + 2b) / 3
        if f(c) < f(d)
            b = d
        else
            a = c
        end
    end
    return (a + b) / 2
end

function numgrad(f, x, h=1e-7)
    n = length(x)
    g = zeros(n)
    for i in 1:n
        xp = copy(x)
        xm = copy(x)
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2h)
    end
    return g
end

function line_search_bounds(x, d, ranges)
    n = length(x)
    lo, hi = 0.0, 1.0
    for i in 1:n
        if abs(d[i]) < 1e-12
            continue
        end
        r = ranges[i]
        t1 = (r[1] - x[i]) / d[i]
        t2 = (r[2] - x[i]) / d[i]
        t_lo, t_hi = min(t1, t2), max(t1, t2)
        lo = max(lo, t_lo)
        hi = min(hi, t_hi)
    end
    lo = max(lo, 0.0)
    return lo, hi
end

function fletcher_reeves(g, grad_g, x0, ranges, tol=1e-8, max_iter=200)
    n = length(x0)
    x = copy(x0)
    gx = grad_g(x)
    d = -gx
    g_prev_norm_sq = dot(gx, gx)
    for _ in 1:max_iter
        if norm(gx) < tol
            break
        end
        lo, hi = line_search_bounds(x, d, ranges)
        if hi <= lo + 1e-10
            break
        end
        phi(alpha) = g(x + alpha * d)
        alpha = minimize_1d(phi, lo, hi)
        x = x + alpha * d
        for i in 1:n
            x[i] = clamp(x[i], ranges[i][1], ranges[i][2])
        end
        gx_new = grad_g(x)
        beta = dot(gx_new, gx_new) / (g_prev_norm_sq + 1e-20)
        g_prev_norm_sq = dot(gx_new, gx_new)
        d = -gx_new + beta * d
        gx = gx_new
    end
    return x
end

function fletcher_reeves_minimize(g, ranges, tol=1e-8)
    n = length(ranges)
    x0 = [(r[1] + r[2]) / 2 for r in ranges]
    grad_g = x -> numgrad(g, x)
    return fletcher_reeves(g, grad_g, x0, ranges, tol)
end

function krechti_leva(f, grad_f, x0, eps, n_lambdas; max_iter=10000)
    dim = length(x0)
    m = n_lambdas - 1
    x = copy(x0)
    delta_history = [zeros(dim) for _ in 1:max(1, m)]
    traj = [copy(x)]
    iter_count = 0

    for k in 1:max_iter
        iter_count = k
        if k > 0 && k % 10 == 0
            delta_history = [zeros(dim) for _ in 1:max(1, m)]
        end

        g = grad_f(x)
        if norm(g) < eps
            break
        end

        obj(lambda) = begin
            s = -lambda[1] * g
            for i in 1:m
                s += lambda[i+1] * delta_history[i]
            end
            f(x + s)
        end

        r = max(0.1, min(1.0, norm(g) > 1e-10 ? 1.0 / norm(g) : 1.0))
        ranges = [(0.0, r * 2.0)]
        for _ in 2:n_lambdas
            push!(ranges, (-r * 0.5, r * 0.5))
        end

        lambda_opt = fletcher_reeves_minimize(obj, ranges)

        step = -lambda_opt[1] * g
        for i in 1:m
            step += lambda_opt[i+1] * delta_history[i]
        end
        x_new = x + step

        for i in m:-1:2
            delta_history[i] = delta_history[i-1]
        end
        delta_history[1] = x_new - x

        x = x_new
        push!(traj, copy(x))

        if norm(delta_history[1]) < eps
            break
        end
    end

    return x, f(x), traj, iter_count
end
function draw_trajectory_2d(f, traj, title_str, filename; lw=4)
    xs = [p[1] for p in traj]
    ys = [p[2] for p in traj]
    fv = [f(p) for p in traj]
    rng = max(maximum(fv) - minimum(fv), 1e-10)
    span_x = maximum(xs) - minimum(xs)
    span_y = maximum(ys) - minimum(ys)
    pad = 0.2 * max(span_x, span_y, 0.5)
    xlim = (minimum(xs) - pad, maximum(xs) + pad)
    ylim = (minimum(ys) - pad, maximum(ys) + pad)
    xg = range(xlim[1], xlim[2], length=80)
    yg = range(ylim[1], ylim[2], length=80)
    plt = contour(xg, yg, (x, y) -> f([x, y]), levels=20, color=cgrad(:viridis, alpha=0.6), colorbar=true, legend=false, xlabel="x_1", ylabel="x_2", title=title_str, xlims=xlim, ylims=ylim)
    plot!(plt, xs, ys, color=:white, lw=lw+2, label="")
    for i in 1:length(traj)-1
        t = (fv[i] - minimum(fv)) / rng
        col = RGB(0.1 + 0.8*(1-t), 0.2 + 0.6*(1-t), 0.9 - 0.7*t)
        plot!(plt, xs[i:i+1], ys[i:i+1], color=col, lw=lw, label="")
    end
    scatter!(plt, [xs[1]], [ys[1]], color=:lime, ms=12, marker=:star5, label="старт")
    scatter!(plt, [xs[end]], [ys[end]], color=:red, ms=12, marker=:diamond, label="конец")
    savefig(plt, filename)
end
# --- 10. Запуск ---
eps_tol = 1e-6
all_results = []
funcs = [
    (name="Розенброк", f=rosenbrock, grad_f=grad_rosenbrock,
     x0=[0.0, 0.0], optimum=[1.0, 1.0], xl=(-0.5, 1.5), yl=(-0.5, 1.5)),
    (name="Швефель", f=schwefel, grad_f=grad_schwefel,
     x0=[350.0, 400.0], optimum=[420.9687, 420.9687], xl=(200.0, 550.0), yl=(200.0, 550.0)),
# Растригин: стартуем из [0.5, 0.5] — в притяжении глобального минимума [0, 0]
    (name="Растригин", f=rastrigin, grad_f=grad_rastrigin,
     x0=[0.5, 0.5], optimum=[0.0, 0.0], xl=(-2.0, 2.0), yl=(-2.0, 2.0)),
]
for fn in funcs
    # Полный МК: оптимизация λ методом Полака-Рибьера
    x_mc, fv_mc, traj_mc, it_mc = mille_contrell(fn.f, fn.grad_f, fn.x0, eps_tol)
    # МК с λ₂=0: одномерное золотое сечение по λ₁
    x_gd, fv_gd, traj_gd, it_gd = mille_contrell(fn.f, fn.grad_f, fn.x0, eps_tol; zero_lambda1=true)
    push!(all_results, (fn.name, "МК (Полак-Рибьер)", it_mc, last(fv_mc), x_mc))
    push!(all_results, (fn.name, "МК (λ₂=0)", it_gd, last(fv_gd), x_gd))
    fig = build_figure(fn.name, fn.f, fn.xl, fn.yl,
                       traj_mc, traj_gd, x_mc, x_gd, fn.optimum)
    savefig(fig, "let4/$(lowercase(fn.name)).png")
end
print_table(all_results)
println("\n✓ Все файлы сохранены в let4/")

eps = 1e-6
n_lambdas_list = [2, 3, 5, 10, 100]
x0 = [320.0, 350.0]
for n in n_lambdas_list
    x, fv, traj, it = krechti_leva(schwefel, grad_schwefel, x0, eps, n)
    println("n = ", n, ":")
    println(" x* = ", round.(x, digits=4))
    println(" f(x*) = ", round(fv, digits=10))
    println(" Итерации = ", it)
    title_str = "Швефель 2D — n=$n λ, итераций=$it"
    filename = joinpath("let4", "schwefel_n$(n)_trajectory.png")
    draw_trajectory_2d(schwefel, traj, title_str, filename)
end

