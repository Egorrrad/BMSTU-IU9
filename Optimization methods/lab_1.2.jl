
using Plots
using LinearAlgebra
using Printf
gr()

struct TestFunction
    name::String
    func::Function
    bounds::Vector{Tuple{Float64,Float64}}
    global_optima::Vector{Float64}
    global_optimum_value::Float64
    local_optima::Vector{Vector{Float64}}
    description::String
end

function get_test_functions()
    return Dict(
        "rosenbrock" => TestFunction("Rosenbrock (Розенброк)",
            x -> (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2,
            [(-2.5, 2.5), (-2.5, 2.5)], [1.0, 1.0], 0.0, Vector{Float64}[],
            "f(x) = (1-x₁)² + 100(x₂-x₁²)²"),

        "rastrigin" => TestFunction("Rastrigin (Растригин)",
            x -> 20 + sum(x[i]^2 - 10*cos(2*π*x[i]) for i in 1:length(x)),
            [(-5.0, 5.0), (-5.0, 5.0)], [0.0, 0.0], 0.0,
            [[1.0,0.0],[-1.0,0.0],[0.0,1.0],[0.0,-1.0]],
            "f(x) = 20 + Σ(xᵢ² - 10cos(2πxᵢ))"),

        "schwefel" => TestFunction("Schwefel (Швефель)",
            x -> 418.9829*length(x) - sum(x[i]*sin(sqrt(abs(x[i]))) for i in 1:length(x)),
            [(-10.0, 10.0), (-10.0, 10.0)], [420.9687, 420.9687], 0.0, Vector{Float64}[],
            "f(x) = 418.9829n - Σxᵢ sin(√|xᵢ|)")
    )
end

function plot_test_function_3d(tf::TestFunction; resolution=80)
    xr = range(tf.bounds[1][1], tf.bounds[1][2], length=resolution)
    yr = range(tf.bounds[2][1], tf.bounds[2][2], length=resolution)
    Z = [tf.func([x,y]) for y in yr, x in xr]

    p = surface(xr, yr, Z, color=:viridis, alpha=0.85, camera=(30,60),
                xlabel="x", ylabel="y", zlabel="f(x,y)",
                title="$(tf.name) — 3D", size=(700,600))

    opt = tf.global_optima
    if length(opt)==2 && all(tf.bounds[i][1] ≤ opt[i] ≤ tf.bounds[i][2] for i=1:2)
        scatter3d!(p, [opt[1]], [opt[2]], [tf.global_optimum_value],
                   marker=:star, markersize=10, color=:red, label="Глобальный минимум")
    end
    return p
end

function plot_test_function_contour(tf::TestFunction; resolution=150)
    xr = range(tf.bounds[1][1], tf.bounds[1][2], length=resolution)
    yr = range(tf.bounds[2][1], tf.bounds[2][2], length=resolution)
    Z = [tf.func([x,y]) for y in yr, x in xr]

    p = contour(xr, yr, Z, fill=true, color=:viridis, levels=40,
                xlabel="x", ylabel="y", title="$(tf.name) — 2D контур", size=(700,600))

    opt = tf.global_optima
    if length(opt)==2 && all(tf.bounds[i][1] ≤ opt[i] ≤ tf.bounds[i][2] for i=1:2)
        scatter!(p, [opt[1]], [opt[2]],
                 marker=:star, markersize=12, color=:red, label="Глобальный минимум")
    end
    return p
end

function exploratory_search(f, x, delta)
    n = length(x)
    x_new = copy(x)
    points = Vector{Vector{Float64}}()
    for i in 1:n
        start = f(x_new)
        x_plus  = copy(x_new); x_plus[i]  += delta[i]
        x_minus = copy(x_new); x_minus[i] -= delta[i]
        push!(points, copy(x_plus), copy(x_minus))
        if f(x_plus) < start; x_new = x_plus
        elseif f(x_minus) < start; x_new = x_minus end
    end
    return x_new, points
end

function hooke_jeeves_variant_a(f, x0, eps=1e-6, delta0=0.1)
    delta = fill(delta0, length(x0))
    x = copy(x0)
    trajectory = [copy(x)]
    search_points = [copy(x)]
    max_points = []

    while maximum(delta) > eps
        x_prev, f_prev = copy(x), f(x)
        x_exp, dir_points = exploratory_search(f, x, delta)
        append!(search_points, dir_points)

        if f(x_exp) >= f(x)
            push!(max_points, copy(x))
            delta ./= 2
            continue
        end

        d = x_exp - x
        step = norm(d)
        x = (step > eps && f(x_exp + d) < f(x_exp)) ? x_exp + d : x_exp

        push!(trajectory, copy(x))
        if norm(x - x_prev) < eps || abs(f(x)-f_prev) < eps; break; end
    end
    return x, trajectory, search_points, max_points
end

function hooke_jeeves_variant_b(f, x0, eps=1e-6, delta0=0.1)
    delta = fill(delta0, length(x0))
    x = copy(x0)
    trajectory = [copy(x)]
    search_points = [copy(x)]
    max_points = []

    while maximum(delta) > eps
        x_prev, f_prev = copy(x), f(x)
        x_exp, dir_points = exploratory_search(f, x, delta)
        append!(search_points, dir_points)

        if f(x_exp) >= f(x)
            push!(max_points, copy(x))
            delta ./= 2
            continue
        end

        d = x_exp - x
        step = norm(d)
        if step > eps
            dir_norm = d / step
            x_cur, f_cur = copy(x_exp), f(x_exp)
            while true
                x_next = x_cur + dir_norm * step
                f_next = f(x_next)
                if f_next < f_cur
                    x_cur, f_cur = x_next, f_next
                    push!(trajectory, copy(x_cur))
                else break end
            end
            x_rollback = x_cur - dir_norm * step / 2
            x = f(x_rollback) < f_cur ? x_rollback : x_cur
        else
            x = x_exp
        end

        push!(trajectory, copy(x))
        if norm(x - x_prev) < eps || abs(f(x)-f_prev) < eps; break; end
    end
    return x, trajectory, search_points, max_points
end

function plot_and_save_combined(tf, x0, result, trajectory, search_points, max_points, variant, func_name)
    f = tf.func

    # Базовые графики
    p3d = plot_test_function_3d(tf)
    p2d = plot_test_function_contour(tf)

    tx = [pt[1] for pt in trajectory]
    ty = [pt[2] for pt in trajectory]
    tz = [f(pt) for pt in trajectory]

    plot!(p3d, tx, ty, tz, linecolor=:lime, linewidth=4, label="Релаксационная последовательность")
    scatter3d!(p3d, tx, ty, tz, color=:green, markersize=5, label="Убывание")

    if !isempty(search_points)
        scatter3d!(p3d, [pt[1] for pt in search_points], [pt[2] for pt in search_points],
                   [f(pt) for pt in search_points], color=:blue, markersize=3, label="Поиск направления")
    end
    if !isempty(max_points)
        scatter3d!(p3d, [pt[1] for pt in max_points], [pt[2] for pt in max_points],
                   [f(pt) for pt in max_points], color=:red, markersize=6, label="Уменьшение шага")
    end
    scatter3d!(p3d, [x0[1]],[x0[2]],[f(x0)], color=:yellow, markersize=10, label="Старт")
    scatter3d!(p3d, [result[1]],[result[2]],[f(result)], color=:magenta, markersize=10, label="Финиш")

    plot!(p2d, tx, ty, color=:lime, linewidth=4, label="Релаксационная последовательность")
    scatter!(p2d, tx, ty, color=:green, markersize=5, label="Убывание")

    if !isempty(search_points)
        scatter!(p2d, [pt[1] for pt in search_points], [pt[2] for pt in search_points],
                 color=:blue, markersize=3, label="Поиск направления")
    end
    if !isempty(max_points)
        scatter!(p2d, [pt[1] for pt in max_points], [pt[2] for pt in max_points],
                 color=:red, markersize=6, label="Уменьшение шага")
    end
    scatter!(p2d, [x0[1]],[x0[2]], color=:yellow, markersize=10, label="Старт")
    scatter!(p2d, [result[1]],[result[2]], color=:magenta, markersize=10, label="Финиш")

    combined = plot(p3d, p2d, layout=(1,2), size=(1450, 680),
                    plot_title="Метод Хука-Дживса (Вариант $variant) — $func_name")

    display(combined)

    filename = "hooke_jeeves_$(func_name)_variant_$(variant)_combined.png"
    savefig(combined, filename)
    println("   Сохранён комбинированный график: $filename")
end

function run_test(tf::TestFunction, x0, variant)
    println("\n" * "="^95)
    println("ТЕСТ: $(tf.name) | Вариант $variant | x0 = $x0")
    println("="^95)

    if variant == "a"
        result, traj, search, maxp = hooke_jeeves_variant_a(tf.func, x0)
    else
        result, traj, search, maxp = hooke_jeeves_variant_b(tf.func, x0)
    end

    println("Результат: $result")
    println("f(result) = $(round(tf.func(result), digits=8))")
    println("Точек в релаксационной последовательности: $(length(traj))")

    func_name = split(tf.name, " (")[1]
    plot_and_save_combined(tf, x0, result, traj, search, maxp, variant, func_name)

    return length(traj)
end

function compare(tf, x0)
    println("\nСРАВНЕНИЕ ВАРИАНТОВ ДЛЯ: $(tf.name)")
    len_a = run_test(tf, x0, "a")
    len_b = run_test(tf, x0, "b")
    println("\nВАРИАНТ А  →  $len_a точек")
    println("ВАРИАНТ Б  →  $len_b точек")
end


tf_dict = get_test_functions()

compare(tf_dict["rosenbrock"], [-2.0, -2.0])
compare(tf_dict["rastrigin"],   [0.5, 0.5])
compare(tf_dict["schwefel"],    [0.5, 0.5])


