using Plots
using Random
using BenchmarkTools
using Printf
plotlyjs()

function target_function_1(x)
    return 5 - 24*x + 17*x^2 - (11/3)*x^3 + (1/4)*x^4
end

function target_function_2(x)
    return x^2 - 10*cos(2π*x) + 10
end

function geometric_temperature(T0, k, α)
    return T0 * α^k
end

function logarithmic_temperature(T0, k)
    return T0 / log(1 + k)
end

function fast_temperature(T0, k)
    return T0 / k
end

function simulated_annealing(f, x_min, x_max, T0, temp_schedule, schedule_name;
                              max_iter=500, α=0.95)
    Random.seed!(42)
    x_current = x_min + (x_max - x_min) * rand()
    f_current = f(x_current)

    trajectory = [(x_current, f_current)]

    for k in 1:max_iter
        if schedule_name == "geometric"
            T = temp_schedule(T0, k, α)
        else
            T = temp_schedule(T0, k)
        end

        x_new = x_current + (rand() - 0.5) * (x_max - x_min) * 0.2
        x_new = clamp(x_new, x_min, x_max)
        f_new = f(x_new)

        ΔE = f_new - f_current

        if ΔE < 0 || rand() < exp(-ΔE / T)
            x_current = x_new
            f_current = f_new
        end

        if k % 5 == 0 || k == max_iter
            push!(trajectory, (x_current, f_current))
        end
    end

    return x_current, f_current, trajectory
end

function optimize_and_plot(f, f_name, x_min, x_max, func_num)
    println("Функция: $f_name")
    println("Область: [$x_min, $x_max]")
    println()

    T0       = 100.0
    max_iter = 500
    α        = 0.95

    results      = Dict()
    trajectories = Dict()

    println("Геометрическое охлаждение (T = T0*α^k, α = $α):")
    time_geo = @elapsed begin
        x_geo, f_geo, traj_geo = simulated_annealing(
            f, x_min, x_max, T0, geometric_temperature, "geometric";
            max_iter=max_iter, α=α
        )
    end
    results["geometric"]      = (x_geo, f_geo, time_geo)
    trajectories["geometric"] = traj_geo
    @printf("Время: %.6f с\n", time_geo)
    @printf("Результат: x = %.6f, f(x) = %.6f\n", x_geo, f_geo)

    println("\nЛогарифмическое охлаждение (T = T0/ln(1+k)):")
    time_log = @elapsed begin
        x_log, f_log, traj_log = simulated_annealing(
            f, x_min, x_max, T0, logarithmic_temperature, "logarithmic";
            max_iter=max_iter
        )
    end
    results["logarithmic"]      = (x_log, f_log, time_log)
    trajectories["logarithmic"] = traj_log
    @printf("Время: %.6f с\n", time_log)
    @printf("Результат: x = %.6f, f(x) = %.6f\n", x_log, f_log)

    println("\nБыстрое охлаждение (T = T0/k):")
    time_fast = @elapsed begin
        x_fast, f_fast, traj_fast = simulated_annealing(
            f, x_min, x_max, T0, fast_temperature, "fast";
            max_iter=max_iter
        )
    end
    results["fast"]      = (x_fast, f_fast, time_fast)
    trajectories["fast"] = traj_fast
    @printf("Время: %.6f с\n", time_fast)
    @printf("Результат: x = %.6f, f(x) = %.6f\n", x_fast, f_fast)

    x_plot = range(x_min, x_max, length=500)
    y_plot = f.(x_plot)

    p = plot(x_plot, y_plot;
        label="f(x)", linewidth=2, color=:black,
        title="Оптимизация: $f_name",
        xlabel="x", ylabel="f(x)",
        legend=:topright, grid=true, gridalpha=0.3,
        size=(900, 550), margin=8Plots.mm)

    x_traj_geo = [t[1] for t in trajectories["geometric"]]
    y_traj_geo = [t[2] for t in trajectories["geometric"]]
    plot!(p, x_traj_geo, y_traj_geo;
        label="Геометрическое", color=:blue,
        linewidth=1.5, alpha=0.7, linestyle=:dash)
    scatter!(p, x_traj_geo, y_traj_geo;
        color=:blue, markersize=3, alpha=0.7, label="")

    x_traj_fast = [t[1] for t in trajectories["fast"]]
    y_traj_fast = [t[2] for t in trajectories["fast"]]
    plot!(p, x_traj_fast, y_traj_fast;
        label="Быстрое", color=:gold,
        linewidth=1.5, alpha=0.7, linestyle=:dash)
    scatter!(p, x_traj_fast, y_traj_fast;
        color=:gold, markersize=3, alpha=0.7, label="")

    x_traj_log = [t[1] for t in trajectories["logarithmic"]]
    y_traj_log = [t[2] for t in trajectories["logarithmic"]]
    plot!(p, x_traj_log, y_traj_log;
        label="Логарифмическое", color=:red,
        linewidth=1.5, alpha=0.7, linestyle=:dash)
    scatter!(p, x_traj_log, y_traj_log;
        color=:red, markersize=3, alpha=0.7, label="")

    scatter!(p, [x_traj_geo[1]], [y_traj_geo[1]];
        label="Старт", color=:green,
        markersize=10, markershape=:star5, markerstrokewidth=0)

    scatter!(p, [x_traj_geo[end]], [y_traj_geo[end]];
        label="Геом. минимум", color=:blue,
        markersize=8, markershape=:circle, markerstrokewidth=1,
        markerstrokecolor=:black)
    scatter!(p, [x_traj_fast[end]], [y_traj_fast[end]];
        label="Быстр. минимум", color=:gold,
        markersize=8, markershape=:circle, markerstrokewidth=1,
        markerstrokecolor=:black)
    scatter!(p, [x_traj_log[end]], [y_traj_log[end]];
        label="Лог. минимум", color=:red,
        markersize=8, markershape=:circle, markerstrokewidth=1,
        markerstrokecolor=:black)

    display(p)
    savefig(p, "let8/optimization_func$(func_num).png")
    println("\nСохранено: let8/optimization_func$(func_num).png")

    return results, trajectories
end

function plot_temperature_profiles()
    T0       = 75.0
    max_iter = 2000
    α        = 0.55

    k_values  = 1:max_iter
    geo_temp  = [geometric_temperature(T0, k, α)  for k in k_values]
    log_temp  = [logarithmic_temperature(T0, k)    for k in k_values]
    fast_temp = [fast_temperature(T0, k)            for k in k_values]

    p = plot(k_values, geo_temp;
        label="Геометрическое (α=$α)", linewidth=2, color=:blue,
        xlabel="Итерация k", ylabel="T(k)",
        title="Схемы охлаждения", legend=:topright,
        size=(900, 500), margin=8Plots.mm)
    plot!(p, k_values, log_temp;  label="Логарифмическое", linewidth=2, color=:red)
    plot!(p, k_values, fast_temp; label="Быстрое (T0/k)",  linewidth=2, color=:gold)

    display(p)
    savefig(p, "let8/temperature_profiles.png")
    println("Сохранено: let8/temperature_profiles.png")
end

function main()
    mkpath("let8")

    plot_temperature_profiles()

    results1, _ = optimize_and_plot(
        target_function_1,
        "f(x) = 5 - 24x + 17x² - (11/3)x³ + (1/4)x⁴",
        0.0, 7.0, 1
    )

    results2, _ = optimize_and_plot(
        target_function_2,
        "f(x) = x² - 10cos(2πx) + 10",
        0.0, 7.0, 2
    )

    println("\nВремя выполнения — функция 1:")
    @printf("  Геометрическое:  %.6f с\n", results1["geometric"][3])
    @printf("  Быстрое:         %.6f с\n", results1["fast"][3])
    @printf("  Логарифмическое: %.6f с\n", results1["logarithmic"][3])

    println("\nВремя выполнения — функция 2:")
    @printf("  Геометрическое:  %.6f с\n", results2["geometric"][3])
    @printf("  Быстрое:         %.6f с\n", results2["fast"][3])
    @printf("  Логарифмическое: %.6f с\n", results2["logarithmic"][3])

    times1   = [results1["geometric"][3], results1["fast"][3], results1["logarithmic"][3]]
    times2   = [results2["geometric"][3], results2["fast"][3], results2["logarithmic"][3]]
    names    = ["Геометрическое", "Быстрое", "Логарифмическое"]
    println("\nПобедитель функция 1: $(names[argmin(times1)]) ($(round(minimum(times1), digits=6)) с)")
    println("Победитель функция 2: $(names[argmin(times2)]) ($(round(minimum(times2), digits=6)) с)")

    println("\nТочность (глобальный минимум f(x)):")
    println("  Функция 1 — аналитически: x ≈ 3.5, f(x) ≈ -8.5")
    @printf("  Геометрическое:  f(x) = %.6f\n", results1["geometric"][2])
    @printf("  Быстрое:         f(x) = %.6f\n", results1["fast"][2])
    @printf("  Логарифмическое: f(x) = %.6f\n", results1["logarithmic"][2])

    println("\n  Функция 2 — аналитически: x = 0,1,2,3..., f(x) = 0")
    @printf("  Геометрическое:  f(x) = %.6f\n", results2["geometric"][2])
    @printf("  Быстрое:         f(x) = %.6f\n", results2["fast"][2])
    @printf("  Логарифмическое: f(x) = %.6f\n", results2["logarithmic"][2])
end

main()

import Pkg; Pkg.add("BenchmarkTools")
