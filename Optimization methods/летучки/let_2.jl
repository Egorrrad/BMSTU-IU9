
using Plots
using LinearAlgebra

gr()

function rosenbrock(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function rastrigin(x)
    n = length(x)
    return 20 + sum(x[i]^2 - 10 * cos(2 * π * x[i]) for i in 1:n)
end

function schwefel(x)
    n = length(x)
    return 418.9829 * n - sum(x[i] * sin(sqrt(abs(x[i]))) for i in 1:n)
end

function create_initial_simplex(x0, n, alpha=1.0)
    simplex = [copy(x0)]
    q = (sqrt(n+1) - 1) / (n * sqrt(2))
    p = q + 1 / sqrt(2)
    for i in 1:n
        x_new = copy(x0)
        for j in 1:n
            if j == i
                x_new[j] += alpha * p
            else
                x_new[j] += alpha * q
            end
        end
        push!(simplex, x_new)
    end
    return simplex
end

function simplex_method(f, x0; eps=1e-6, alpha=1.0, max_iterations=5000, verbose=false)
    n = length(x0)
    simplex = create_initial_simplex(x0, n, alpha)
    trajectory = [copy(x0)]
    all_simplexes = [copy(simplex)]
    iteration = 0
    last_best_value = f(simplex[argmin(f.(simplex))])
    no_improvement_count = 0


    while iteration < max_iterations
        iteration += 1
        sorted_indices = sortperm(f.(simplex))
        simplex = simplex[sorted_indices]
        centroid = sum(simplex[1:end-1]) / n
        x_best, x_worst = simplex[1], simplex[end]
        f_best, f_worst = f(x_best), f(x_worst)
        current_best_value = f_best

        if abs(current_best_value - last_best_value) < eps
            no_improvement_count += 1
            if no_improvement_count > 15
                break
            end
        else
            no_improvement_count = 0
            last_best_value = current_best_value
        end

        if maximum(norm.(simplex .- Ref(centroid))) < eps
            break
        end

        x_r = centroid + 1.0 * (centroid - x_worst)
        f_r = f(x_r)

        if f_r < f_best
            x_e = centroid + 2.0 * (centroid - x_worst)
            f_e = f(x_e)
            if f_e < f_r
                simplex[end] = x_e
                push!(trajectory, copy(x_e))
            else
                simplex[end] = x_r
                push!(trajectory, copy(x_r))
            end
        elseif f_r < f_worst
            simplex[end] = x_r
            push!(trajectory, copy(x_r))
        else
            x_c = if f_r < f_worst
                centroid + 0.5 * (x_r - centroid)
            else
                centroid - 0.5 * (centroid - x_worst)
            end
            f_c = f(x_c)

            if f_c < min(f_r, f_worst)
                simplex[end] = x_c
                push!(trajectory, copy(x_c))
            else
                for i in 2:n+1
                    simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                end
                push!(trajectory, copy(x_best))
            end
        end
        push!(all_simplexes, copy(simplex))
    end

    best_idx = argmin(f.(simplex))
    result = simplex[best_idx]
    return result, trajectory, all_simplexes
end


function save_combined_plot(f, x0, result, trajectory, all_simplexes, title_str, x_range, y_range, n; filename="combined.png")    
    x_grid = range(x_range[1], x_range[2], length=100)
    y_grid = range(y_range[1], y_range[2], length=100)
    Z = [f([a, b]) for a in x_grid, b in y_grid]

    
    p2d = contour(x_grid, y_grid, Z,
                  color=:thermal, levels=25,
                  xlabel="x₁", ylabel="x₂",
                  title="$title_str\n(контур)",
                  legend=:topleft, size=(600,500))

    colors = palette(:tab10)

    for (idx, simp) in enumerate(all_simplexes)
        if length(simp) >= 3
            col = colors[mod1(idx, length(colors))]
            pts = [simp[1], simp[2], simp[3], simp[1]]
            plot!(p2d, [p[1] for p in pts], [p[2] for p in pts],
                  color=col, linewidth=1.5,
                  label=(idx==1 ? "Симплексы" : ""), alpha=0.7)
        end
    end

    if !isempty(trajectory)
        step = max(1, length(trajectory) ÷ 500)
        traj_f = trajectory[1:step:end]
        scatter!(p2d, [p[1] for p in traj_f], [p[2] for p in traj_f],
                 color=:green, markersize=2.5, label="Траектория")
    end

    scatter!(p2d, [x0[1]], [x0[2]], color=:yellow, markersize=9, label="Старт")
    scatter!(p2d, [result[1]], [result[2]], color=:purple, markersize=9, label="Финиш")
    plot!(p2d, colorbar=true)

    p3d = surface(x_grid, y_grid, Z,
                  color=:thermal, alpha=0.5,
                  xlabel="x₁", ylabel="x₂", zlabel="f(x)",
                  title="$title_str\n(поверхность)",
                  legend=true, size=(600,500))

    for (idx, simp) in enumerate(all_simplexes)
        if length(simp) >= 3
            col = colors[mod1(idx, length(colors))]
            pts = [simp[1], simp[2], simp[3], simp[1]]
            plot!(p3d,
                  [p[1] for p in pts], [p[2] for p in pts], [f(p) for p in pts],
                  linecolor=col, linewidth=1.5,
                  label=(idx==1 ? "Симплексы" : ""), alpha=0.6)
        end
    end

    if length(trajectory) > 1
        step = max(1, length(trajectory) ÷ 500)
        traj_f = trajectory[1:step:end]
        scatter3d!(p3d,
                   [p[1] for p in traj_f],
                   [p[2] for p in traj_f],
                   [f(p) for p in traj_f],
                   color=:green, markersize=2, label="Траектория")
    end

    scatter3d!(p3d, [x0[1]], [x0[2]], [f(x0)], color=:yellow, markersize=8, label="Старт")
    scatter3d!(p3d, [result[1]], [result[2]], [f(result)], color=:purple, markersize=8, label="Финиш")

    combined = plot(p2d, p3d, layout=(1,2), size=(1200, 600),
                    plot_title=title_str, plot_titlefontsize=14)

    savefig(combined, filename)
    println("Сохранён комбинированный график: $filename")
end


function test_simplex(f, f_name, x0, x_range, y_range, n; show_plots=true)
    println("\n" * "═"^60)
    println("Тестирование функции: $f_name")
    println("Начальная точка: $x0")
    println("═"^60)

    result, trajectory, all_simplexes = simplex_method(f, x0)

    println("Результат: ", round.(result, digits=6))
    println("Значение функции: ", f(result))
    println("Количество итераций: ", length(all_simplexes))
    println("Точек в траектории: ", length(trajectory))

    title_str = "Метод простого симплекса — $f_name"

    filename = "optimization_result_$f_name.png"
    save_combined_plot(f, x0, result, trajectory, all_simplexes,
                       title_str, x_range, y_range, n; filename=filename)

    
    if show_plots
        plt_3d = plot_simplex_3d(f, x0, result, trajectory, all_simplexes, title_str, x_range, y_range, n)
        plt_2d = plot_simplex_2d(f, x0, result, trajectory, all_simplexes, title_str, x_range, y_range, n)
        display(plt_3d)
        display(plt_2d)
    end

    return result, trajectory
end

println("Метод простого симплекса: Тестирование на трёх функциях\n")

println("=== Тест 1: Розенброк ===")
test_simplex(rosenbrock, "Розенброк", [-2.0, -2.0], (-2.5, 2.5), (-2.5, 2.5), 2)

println("\n=== Тест 2: Растригин ===")
test_simplex(rastrigin, "Растригин", [0.5, 0.5], (-5.5, 5.5), (-5.5, 5.5), 2)

println("\n=== Тест 3: Швефель ===")
test_simplex(schwefel, "Швефель", [3.0, 4.0], (-10, 10), (-10, 10), 2)
