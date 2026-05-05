using Plots
using Random
using Statistics

const OUTPUT_DIR = "let7"
isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)

function generate_initial_population()
    return [2.0, 3.0, 4.0, 5.0]
end

function mutate(x, mutation_rate, x_range)
    r1, r2 = x_range
    if rand() < mutation_rate
        return rand() * (r2 - r1) + r1
    end
    return x
end

function select(population, fitness, number_of_parents)
    scores = [fitness(x) for x in population]
    sorted_idx = sortperm(scores)
    return population[sorted_idx[1:number_of_parents]]
end

function binary_single_point_crossover(parent1, parent2, x_range; crossover_point=nothing)
    p1_bits = bitstring(Float64(parent1))
    p2_bits = bitstring(Float64(parent2))

    cp = isnothing(crossover_point) ? rand(1:length(p1_bits)-1) : crossover_point

    c1_bits = p1_bits[1:cp] * p2_bits[cp+1:end]
    c2_bits = p2_bits[1:cp] * p1_bits[cp+1:end]

    child1 = parent1
    child2 = parent2

    try
        child1 = reinterpret(Float64, parse(UInt64, c1_bits, base=2))
        child2 = reinterpret(Float64, parse(UInt64, c2_bits, base=2))
    catch
    end

    if isnan(child1) || isinf(child1)
        child1 = parent1
    end
    if isnan(child2) || isinf(child2)
        child2 = parent2
    end

    child1 = clamp(child1, x_range[1], x_range[2])
    child2 = clamp(child2, x_range[1], x_range[2])

    return child1, child2, cp
end

function crossover(parents, number_of_children, crossover_type, x_range;
                   crossover_point_mode="random", fixed_point=nothing)
    children = Float64[]
    crossover_points = Int[]

    if crossover_type == "average"
        children = [(rand(parents) + rand(parents)) / 2 for _ in 1:number_of_children]
        crossover_points = zeros(Int, number_of_children)

    elseif crossover_type == "binary"
        while length(children) < number_of_children
            p1, p2 = rand(parents), rand(parents)

            if crossover_point_mode == "fixed"
                c1, c2, cp = binary_single_point_crossover(p1, p2, x_range; crossover_point=fixed_point)
            elseif crossover_point_mode == "random"
                c1, c2, cp = binary_single_point_crossover(p1, p2, x_range)
            else
                error("Неизвестный режим точки кроссинговера: $crossover_point_mode")
            end

            push!(children, c1)
            push!(crossover_points, cp)

            if length(children) < number_of_children
                push!(children, c2)
                push!(crossover_points, cp)
            end
        end
    else
        error("Неизвестный тип кроссинговера: $crossover_type")
    end

    return children, crossover_points
end

function genetic_algorithm(
    fitness,
    generations,
    x_range,
    mutation_rate;
    crossover_type="binary",
    crossover_point_mode="random",
    seed=123,
    make_gif=true,
    gif_name="ga.gif"
)
    Random.seed!(seed)

    population = generate_initial_population()
    initial_population = copy(population)
    population_size = length(population)

    if population_size < 2
        error("Размер популяции должен быть не меньше 2")
    end

    best_idx = argmin([fitness(x) for x in population])
    best_solution = population[best_idx]
    best_score = fitness(best_solution)

    best_score_history = Float64[best_score]
    best_x_history = Float64[best_solution]

    fixed_point = nothing
    if crossover_type == "binary" && crossover_point_mode == "fixed"
        fixed_point = rand(1:63)
    end

    x_vals = range(x_range[1], x_range[2], length=400)
    y_vals = fitness.(x_vals)
    y_min, y_max = minimum(y_vals), maximum(y_vals)
    plot_range = (
        y_min - 0.1 * (y_max - y_min),
        y_max + 0.1 * (y_max - y_min)
    )

    anim = make_gif ? Animation() : nothing

    for generation in 0:generations
        current_scores = [fitness(x) for x in population]
        current_best_idx = argmin(current_scores)
        current_best = population[current_best_idx]
        current_score = current_scores[current_best_idx]

        if current_score < best_score
            best_solution = current_best
            best_score = current_score
        end

        if generation > 0
            push!(best_score_history, best_score)
            push!(best_x_history, best_solution)
        end

        if make_gif
            plt = plot(
                x_vals,
                y_vals,
                linewidth=2,
                xlabel="x",
                ylabel="f(x)",
                title="Поколение $generation | мутация = $mutation_rate | кроссинговер = $crossover_point_mode",
                legend=:topright,
                xlims=x_range,
                ylims=plot_range,
                grid=true,
                size=(950, 600)
            )

            scatter!(
                plt,
                initial_population,
                fitness.(initial_population),
                color=:blue,
                marker=:x,
                markersize=8,
                label="Начальная популяция"
            )

            scatter!(
                plt,
                population,
                fitness.(population),
                color=:lightgreen,
                alpha=0.7,
                markersize=5,
                label="Текущая популяция"
            )

            scatter!(
                plt,
                [current_best],
                [current_score],
                color=:red,
                marker=:circle,
                markersize=8,
                label="Лучшее текущее решение"
            )

            if generation == generations
                scatter!(
                    plt,
                    [best_solution],
                    [best_score],
                    color=:black,
                    marker=:circle,
                    markersize=9,
                    label="Лучшее найденное решение"
                )
            end

            annotate!(
                plt,
                x_range[1] + 0.05 * (x_range[2] - x_range[1]),
                plot_range[1] + 0.08 * (plot_range[2] - plot_range[1]),
                text(
                    "Лучшее: x = $(round(best_solution, digits=4)), f(x) = $(round(best_score, digits=4))",
                    :left,
                    10
                )
            )

            frame(anim)
        end

        if generation == generations
            break
        end

        parents = select(population, fitness, population_size ÷ 2)

        children, crossover_points = crossover(
            parents,
            population_size - length(parents),
            crossover_type,
            x_range;
            crossover_point_mode=crossover_point_mode,
            fixed_point=fixed_point
        )

        children = [mutate(c, mutation_rate, x_range) for c in children]
        population = vcat(parents, children)

        if generation > 0 && generation % 100 == 0
            println("Поколение $generation | кроссинговер = $crossover_point_mode | мутация = $mutation_rate")
            println("лучшее x = ", round(best_solution, digits=6))
            println("лучшее f(x) = ", round(best_score, digits=6))
            println("среднее по популяции = ", round(mean(population), digits=6))
            println("стандартное отклонение = ", round(std(population), digits=6))
            if crossover_type == "binary" && !isempty(crossover_points)
                println("средняя точка кроссинговера = ", round(mean(crossover_points), digits=3))
            end
            println()
        end
    end

    if make_gif
        gif_path = joinpath(OUTPUT_DIR, gif_name)
        gif(anim, gif_path, fps=5)
        println("GIF сохранён: $gif_path")
    end

    return best_solution, best_score, best_score_history, best_x_history
end

function plot_comparison(
    score_hist_fixed,
    score_hist_random,
    x_hist_fixed,
    x_hist_random,
    final_x_fixed,
    final_fx_fixed,
    final_x_random,
    final_fx_random,
    mutation_rate
)
    generations_axis = 0:(length(score_hist_fixed) - 1)

    p1 = plot(
        generations_axis,
        score_hist_fixed,
        linewidth=3,
        label="Фиксированная точка: f(x)",
        xlabel="Поколение",
        ylabel="Лучшее значение f(x)",
        title="Сходимость алгоритма, мутация = $mutation_rate",
        legend=:topright,
        grid=true,
        size=(1000, 800)
    )

    plot!(
        p1,
        generations_axis,
        score_hist_random,
        linewidth=3,
        linestyle=:dash,
        label="Случайная точка: f(x)"
    )

    scatter!(
        p1,
        [generations_axis[end]],
        [final_fx_fixed],
        color=:green,
        markersize=8,
        label="Фиксированная точка: f(x) = $(round(final_fx_fixed, digits=6))"
    )

    scatter!(
        p1,
        [generations_axis[end]],
        [final_fx_random],
        color=:purple,
        markersize=8,
        label="Случайная точка: f(x) = $(round(final_fx_random, digits=6))"
    )

    p2 = plot(
        generations_axis,
        x_hist_fixed,
        linewidth=3,
        label="Фиксированная точка: x",
        xlabel="Поколение",
        ylabel="Координата оптимума x",
        title="Изменение координаты лучшего решения",
        legend=:topright,
        grid=true,
        size=(1000, 800)
    )

    plot!(
        p2,
        generations_axis,
        x_hist_random,
        linewidth=3,
        linestyle=:dash,
        label="Случайная точка: x"
    )

    scatter!(
        p2,
        [generations_axis[end]],
        [final_x_fixed],
        color=:green,
        markersize=8,
        label="Фиксированная точка: x = $(round(final_x_fixed, digits=6))"
    )

    scatter!(
        p2,
        [generations_axis[end]],
        [final_x_random],
        color=:purple,
        markersize=8,
        label="Случайная точка: x = $(round(final_x_random, digits=6))"
    )

    final_plot = plot(p1, p2, layout=(2, 1), size=(1000, 900))

    filename = "comparison_mutation_$(replace(string(mutation_rate), "." => "_")).png"
    filepath = joinpath(OUTPUT_DIR, filename)
    savefig(final_plot, filepath)
    println("Сохранён файл: $filepath")

    return final_plot
end

function plot_summary_by_mutation(mutation_values, final_results_fixed, final_results_random)
    y_fixed = [r[:best_f] for r in final_results_fixed]
    y_random = [r[:best_f] for r in final_results_random]

    labels_x = string.(mutation_values)

    plt = plot(
        1:length(mutation_values), y_fixed,
        linewidth=3,
        marker=:circle,
        markersize=7,
        label="Фиксированная точка: f(x)",
        xlabel="Вероятность мутации",
        ylabel="Итоговое лучшее значение f(x)",
        title="Итоговые значения функции при разных уровнях мутации",
        xticks=(1:length(mutation_values), labels_x),
        legend=:topright,
        grid=true,
        size=(1000, 600)
    )

    plot!(
        plt,
        1:length(mutation_values), y_random,
        linewidth=3,
        linestyle=:dash,
        marker=:square,
        markersize=7,
        label="Случайная точка: f(x)"
    )

    filename = "summary_all_mutations.png"
    filepath = joinpath(OUTPUT_DIR, filename)
    savefig(plt, filepath)
    println("Сохранён файл: $filepath")

    return plt
end

fit(x) = 5 - 24*x + 17*x^2 - (11/3) * x^3 + (1/4) * x^4

generations = 100
x_range = (-2.0, 9.0)
mutation_values = [0.0, 0.3, 0.5]
crossover_type = "binary"
seed = 123

all_comparison_plots = Any[]
final_results_fixed = Dict[]
final_results_random = Dict[]

for mutation_rate in mutation_values
    println("Вероятность мутации = $mutation_rate")

    println("Режим: фиксированная точка кроссинговера")
    gif_fixed = "genetic_algorithm_binary_fixed_mut$(replace(string(mutation_rate), "." => "_")).gif"

    best_fixed, best_f_fixed, score_hist_fixed, x_hist_fixed = genetic_algorithm(
        fit,
        generations,
        x_range,
        mutation_rate;
        crossover_type=crossover_type,
        crossover_point_mode="fixed",
        seed=seed,
        make_gif=true,
        gif_name=gif_fixed
    )

    push!(final_results_fixed, Dict(
        :mutation => mutation_rate,
        :best_x => best_fixed,
        :best_f => best_f_fixed,
        :score_hist => score_hist_fixed,
        :x_hist => x_hist_fixed
    ))

    println("Режим: случайная точка кроссинговера")
    gif_random = "genetic_algorithm_binary_random_mut$(replace(string(mutation_rate), "." => "_")).gif"

    best_random, best_f_random, score_hist_random, x_hist_random = genetic_algorithm(
        fit,
        generations,
        x_range,
        mutation_rate;
        crossover_type=crossover_type,
        crossover_point_mode="random",
        seed=seed,
        make_gif=true,
        gif_name=gif_random
    )

    push!(final_results_random, Dict(
        :mutation => mutation_rate,
        :best_x => best_random,
        :best_f => best_f_random,
        :score_hist => score_hist_random,
        :x_hist => x_hist_random
    ))

    comparison_plot = plot_comparison(
        score_hist_fixed,
        score_hist_random,
        x_hist_fixed,
        x_hist_random,
        best_fixed,
        best_f_fixed,
        best_random,
        best_f_random,
        mutation_rate
    )

    display(comparison_plot)
    push!(all_comparison_plots, comparison_plot)

    println("Вероятность мутации = ", mutation_rate)
    println("Фиксированная точка | x = ", round(best_fixed, digits=6), " | f(x) = ", round(best_f_fixed, digits=6))
    println("Случайная точка     | x = ", round(best_random, digits=6), " | f(x) = ", round(best_f_random, digits=6))
end

println("\nСводка:")
for i in eachindex(mutation_values)
    m = mutation_values[i]

    best_fixed = final_results_fixed[i][:best_x]
    best_f_fixed = final_results_fixed[i][:best_f]

    best_random = final_results_random[i][:best_x]
    best_f_random = final_results_random[i][:best_f]

    println("Мутация = ", m)
    println("Фиксированная точка | x = ", round(best_fixed, digits=6), " | f(x) = ", round(best_f_fixed, digits=6))
    println("Случайная точка     | x = ", round(best_random, digits=6), " | f(x) = ", round(best_f_random, digits=6))
end

summary_plot = plot_summary_by_mutation(mutation_values, final_results_fixed, final_results_random)
display(summary_plot)
