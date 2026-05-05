
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Printf

mkpath("lab6_output")

# --- 1. Целевые функции ---
rosenbrock(x) = sum(100 * (x[i+1] - x[i]^2)^2 + (x[i] - 1)^2 for i in 1:length(x)-1)
schwefel(x)   = 418.9829 * length(x) - sum(xi -> xi * sin(sqrt(abs(xi))), x)
rastrigin(x)  = 10 * length(x) + sum(xi -> xi^2 - 10 * cos(2π * xi), x)

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
        s  = sqrt(abs(xi)) + 1e-20
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
    lambda  = copy(lambda0)
    grad    = numerical_gradient(g, lambda)
    d       = -copy(grad)
    g_prev  = copy(grad)

    for k in 1:maxiter
        norm(grad) < tol && break

        # Одномерный поиск вдоль направления d (золотое сечение)
        phi(alpha) = g(lambda + alpha * d)
        alpha_opt  = golden_section(phi, 0.0, alpha_max)

        lambda = lambda + alpha_opt * d
        grad   = numerical_gradient(g, lambda)

        # Формула Полака-Рибьера: β = (∇g_new · (∇g_new - ∇g_old)) / ‖∇g_old‖²
        delta_g = grad - g_prev
        denom   = dot(g_prev, g_prev)
        beta    = denom < 1e-30 ? 0.0 : dot(grad, delta_g) / denom
        beta    = max(beta, 0.0)   # β⁺ — рестарт при отрицательном β

        # Рестарт каждые n+1 итераций (n=2 для λ)
        k % 3 == 0 && (beta = 0.0)

        d      = -grad + beta * d
        g_prev = copy(grad)
    end

    return lambda
end

# --- 6. Метод Миле-Кантрелла ---
function mille_contrell(f, grad_f, x0, eps; max_iter=10000, zero_lambda1=false)
    n = length(x0)
    x       = copy(x0)
    x_prev  = copy(x0)
    delta_x = zeros(n)
    traj    = [copy(x)]
    fv      = [f(x)]
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
            l_opt  = [l1_opt, 0.0]
        else
            # Двумерная оптимизация по [λ₁, λ₂] методом Полака-Рибьера
            r      = max(0.1, min(1.0, norm(delta_x) / (norm(g) + 1e-10)))
            lambda0 = [r, 0.0]   # начальное приближение
            l_opt  = minimize_polak_ribiere(obj, lambda0)
            # Гарантируем λ₁ ≥ 0 (шаг по антиградиенту не может быть отрицательным)
            l_opt[1] = max(l_opt[1], 0.0)
        end

        x_new   = x - l_opt[1] * g + l_opt[2] * delta_x
        delta_x = x_new - x
        x_prev  = x
        x       = x_new
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

# --- 8. Построение фигуры: 2D контур + 3D поверхность (стиль лаб5) ---
function build_figure(title_str, f, xl, yl, traj_mc, traj_gd, x_opt_mc, x_opt_gd, optimum)
    # Подгоняем область под реальные траектории + истинный оптимум
    all_px = vcat([p[1] for p in traj_mc], [p[1] for p in traj_gd], [optimum[1]])
    all_py = vcat([p[2] for p in traj_mc], [p[2] for p in traj_gd], [optimum[2]])
    span_x = max(maximum(all_px) - minimum(all_px), 1.0)
    span_y = max(maximum(all_py) - minimum(all_py), 1.0)
    pad_x  = 0.15 * span_x
    pad_y  = 0.15 * span_y
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
    plot!(p2, [0], [0]; color=:white,  lw=2, label="МК (Полак-Рибьер)")
    plot!(p2, [0], [0]; color=:tomato, lw=2, label="МК (λ₂=0)")

    # Траектории на 2D
    tx_mc = [p[1] for p in traj_mc]; ty_mc = [p[2] for p in traj_mc]
    tx_gd = [p[1] for p in traj_gd]; ty_gd = [p[2] for p in traj_gd]
    plot!(p2, tx_mc, ty_mc; color=:white,  linewidth=2.5, label="")
    plot!(p2, tx_gd, ty_gd; color=:tomato, linewidth=2.5, label="")

    # Траектории на 3D
    plot!(p3, tx_mc, ty_mc, [f(p) for p in traj_mc]; color=:white,  linewidth=2, label="")
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

# --- 10. Запуск ---
eps_tol = 1e-6
all_results = []

funcs = [
    (name="Розенброк", f=rosenbrock, grad_f=grad_rosenbrock,
     x0=[0.0, 0.0],     optimum=[1.0, 1.0],           xl=(-0.5, 1.5),    yl=(-0.5, 1.5)),
    (name="Швефель",   f=schwefel,   grad_f=grad_schwefel,
     x0=[350.0, 400.0], optimum=[420.9687, 420.9687],  xl=(200.0, 550.0), yl=(200.0, 550.0)),
    # Растригин: стартуем из [0.5, 0.5] — в притяжении глобального минимума [0, 0]
    (name="Растригин", f=rastrigin,  grad_f=grad_rastrigin,
     x0=[0.5, 0.5],     optimum=[0.0, 0.0],            xl=(-2.0, 2.0),    yl=(-2.0, 2.0)),
]

for fn in funcs
    # Полный МК: оптимизация λ методом Полака-Рибьера
    x_mc, fv_mc, traj_mc, it_mc = mille_contrell(fn.f, fn.grad_f, fn.x0, eps_tol)
    # МК с λ₂=0: одномерное золотое сечение по λ₁
    x_gd, fv_gd, traj_gd, it_gd = mille_contrell(fn.f, fn.grad_f, fn.x0, eps_tol; zero_lambda1=true)

    push!(all_results, (fn.name, "МК (Полак-Рибьер)", it_mc, last(fv_mc), x_mc))
    push!(all_results, (fn.name, "МК (λ₂=0)",         it_gd, last(fv_gd), x_gd))

    fig = build_figure(fn.name, fn.f, fn.xl, fn.yl,
                       traj_mc, traj_gd, x_mc, x_gd, fn.optimum)
    savefig(fig, "lab6_output/$(lowercase(fn.name)).png")
end

print_table(all_results)
println("\n✓ Все файлы сохранены в lab6_output/")

