
ENV["GKSwstype"] = "100"

using Random
using Plots
plotlyjs()

mkpath("lab17")

default(
    fontfamily = "Computer Modern",
    legendfontsize = 8,
    guidefontsize = 11,
    tickfontsize = 9,
    titlefontsize = 12,
)

const GRID_STEP = 0.01
const POPULATION_SIZE = 20
const MAX_ITERATIONS_LIMIT = 2000
const ELITISM_COUNT = 1
const TOURNAMENT_SIZE = 2
const CROSSOVER_RATE = 1.0
const BASE_MAX_RESTARTS = 30
const CONVERGENCE_GAP_EPS = 1e-12
const DEFAULT_MUTATION_RATE = 0.03
const FIXED_MUTATION_RATE = 0.03
const MUTATION_RATES_TO_TEST = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20]
const RNG_SEED = 42
const CROSSOVER_POINT_RATIOS = [0.2, 0.4, 0.6, 0.8]
const TRAJECTORY_SAMPLE_LIMIT = 140
const PLOT_GRID_RESOLUTION = 130

rosenbrock_2d(x::Float64, y::Float64) = (1 - x)^2 + 100 * (y - x^2)^2
rastrigin_2d(x::Float64, y::Float64) = 20 + (x^2 - 10 * cos(2 * pi * x)) + (y^2 - 10 * cos(2 * pi * y))
schwefel_2d(x::Float64, y::Float64) = 2 * 418.9829 - (x * sin(sqrt(abs(x))) + y * sin(sqrt(abs(y))))

struct FunctionConfig
    name::String
    slug::String
    objective::Function
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64
    target_min_point::Tuple{Float64, Float64}
    target_tol::Float64
end

const FUNCTION_CONFIGS = [
    FunctionConfig(
        "Rosenbrock",
        "rosenbrock",
        rosenbrock_2d,
        -2.0,
        2.0,
        -2.0,
        2.0,
        (1.0, 1.0),
        1e-9,
    ),
    FunctionConfig(
        "Rastrigin",
        "rastrigin",
        rastrigin_2d,
        -5.12,
        5.12,
        -5.12,
        5.12,
        (0.0, 0.0),
        1e-9,
    ),
    FunctionConfig(
        "Schwefel",
        "schwefel",
        schwefel_2d,
        -500.0,
        500.0,
        -500.0,
        500.0,
        (420.968746, 420.968746),
        1e-2,
    ),
]

struct EncodingConfig
    x_levels::Int
    y_levels::Int
    x_bits::Int
    y_bits::Int
    x_max_chromosome_value::Int
    y_max_chromosome_value::Int
end

objective_at_point(point::Tuple{Float64, Float64}, config::FunctionConfig) =
    config.objective(point[1], point[2])

target_min_value(config::FunctionConfig) =
    objective_at_point(config.target_min_point, config)

function build_encoding(config::FunctionConfig)
    x_levels = Int(round((config.x_max - config.x_min) / GRID_STEP))
    y_levels = Int(round((config.y_max - config.y_min) / GRID_STEP))
    x_bits = ceil(Int, log2(x_levels + 1))
    y_bits = ceil(Int, log2(y_levels + 1))
    return EncodingConfig(
        x_levels,
        y_levels,
        x_bits,
        y_bits,
        (1 << x_bits) - 1,
        (1 << y_bits) - 1,
    )
end

clamp_gene(g::Int, max_level::Int) = clamp(g, 0, max_level)

function encode_coordinate(value::Float64, min_value::Float64, max_level::Int)
    raw = round(Int, (value - min_value) / GRID_STEP)
    return clamp_gene(raw, max_level)
end

decode_coordinate(gene::Int, min_value::Float64, max_level::Int) =
    min_value + clamp_gene(gene, max_level) * GRID_STEP

function encode_point(
    point::Tuple{Float64, Float64},
    config::FunctionConfig,
    encoding::EncodingConfig,
)
    return (
        encode_coordinate(point[1], config.x_min, encoding.x_levels),
        encode_coordinate(point[2], config.y_min, encoding.y_levels),
    )
end

function decode_gene(
    gene::Tuple{Int, Int},
    config::FunctionConfig,
    encoding::EncodingConfig,
)
    return (
        decode_coordinate(gene[1], config.x_min, encoding.x_levels),
        decode_coordinate(gene[2], config.y_min, encoding.y_levels),
    )
end

function objective_at_gene(
    gene::Tuple{Int, Int},
    config::FunctionConfig,
    encoding::EncodingConfig,
)
    point = decode_gene(gene, config, encoding)
    return objective_at_point(point, config)
end

function max_valid_crossover_point(encoding::EncodingConfig)
    return min(encoding.x_bits, encoding.y_bits) - 1
end

function default_crossover_point(encoding::EncodingConfig)
    max_point = max_valid_crossover_point(encoding)
    @assert max_point >= 1 "Недостаточно бит для кроссинговера"
    return max(1, max_point ÷ 2)
end

function build_initial_population(config::FunctionConfig)
    xs = collect(range(config.x_min, config.x_max; length = 5))
    ys = collect(range(config.y_min, config.y_max; length = 4))
    points = Tuple{Float64, Float64}[]
    for y in ys
        for x in xs
            push!(points, (x, y))
        end
    end
    @assert length(points) == POPULATION_SIZE "build_initial_population должен вернуть POPULATION_SIZE точек"
    return points
end

function one_point_crossover_axis(
    g1::Int,
    g2::Int,
    point::Int,
    bits::Int,
    max_level::Int,
)
    @assert 1 <= point <= bits - 1 "Точка кроссинговера вне диапазона"
    low_mask = (1 << point) - 1
    full_mask = (1 << bits) - 1
    high_mask = full_mask ⊻ low_mask
    child1 = (g1 & high_mask) | (g2 & low_mask)
    child2 = (g2 & high_mask) | (g1 & low_mask)
    return clamp_gene(child1, max_level), clamp_gene(child2, max_level)
end

function one_point_crossover_2d(
    g1::Tuple{Int, Int},
    g2::Tuple{Int, Int},
    point::Int,
    encoding::EncodingConfig,
)
    cx1, cx2 = one_point_crossover_axis(
        g1[1], g2[1], point, encoding.x_bits, encoding.x_levels
    )
    cy1, cy2 = one_point_crossover_axis(
        g1[2], g2[2], point, encoding.y_bits, encoding.y_levels
    )
    return (cx1, cy1), (cx2, cy2)
end

function mutate_axis(
    gene::Int,
    mutation_rate::Float64,
    bits::Int,
    max_level::Int,
    rng::AbstractRNG,
)
    g = gene
    for bit in 0:(bits - 1)
        if rand(rng) < mutation_rate
            g = g ⊻ (1 << bit)
        end
    end
    return clamp_gene(g, max_level)
end

function mutate_gene(
    gene::Tuple{Int, Int},
    mutation_rate::Float64,
    encoding::EncodingConfig,
    rng::AbstractRNG,
)
    return (
        mutate_axis(gene[1], mutation_rate, encoding.x_bits, encoding.x_levels, rng),
        mutate_axis(gene[2], mutation_rate, encoding.y_bits, encoding.y_levels, rng),
    )
end

function tournament_select(
    population::Vector{Tuple{Int, Int}},
    config::FunctionConfig,
    encoding::EncodingConfig,
    rng::AbstractRNG,
)
    sampled_indices = rand(rng, 1:length(population), TOURNAMENT_SIZE)
    best_idx = sampled_indices[1]
    best_val = objective_at_gene(population[best_idx], config, encoding)

    for idx in sampled_indices[2:end]
        cur_val = objective_at_gene(population[idx], config, encoding)
        if cur_val < best_val
            best_idx = idx
            best_val = cur_val
        end
    end

    return population[best_idx]
end

function run_ga(
    config::FunctionConfig,
    encoding::EncodingConfig;
    mutation_rate::Float64 = DEFAULT_MUTATION_RATE,
    crossover_point::Int = default_crossover_point(encoding),
    initial_population_points::Vector{Tuple{Float64, Float64}} = build_initial_population(config),
    seed::Int = RNG_SEED,
)
    @assert 0.0 <= mutation_rate <= 1.0 "mutation_rate должен быть в [0, 1]"
    @assert length(initial_population_points) == POPULATION_SIZE "Размер initial_population_points должен совпадать с POPULATION_SIZE"

    max_cp = max_valid_crossover_point(encoding)
    @assert 1 <= crossover_point <= max_cp "Некорректная точка кроссинговера"

    rng = MersenneTwister(seed)
    population = [encode_point(point, config, encoding) for point in initial_population_points]

    population_history = Vector{Vector{Tuple{Float64, Float64}}}()
    best_values_history = Float64[]
    best_point_history = Tuple{Float64, Float64}[]

    function save_generation!(pop::Vector{Tuple{Int, Int}})
        points = [decode_gene(g, config, encoding) for g in pop]
        vals = [objective_at_point(p, config) for p in points]
        push!(population_history, points)
        best_idx = argmin(vals)
        push!(best_values_history, vals[best_idx])
        push!(best_point_history, points[best_idx])
    end

    target_value = target_min_value(config)
    save_generation!(population)

    iterations_done = 0
    target_reached = best_values_history[end] <= target_value + config.target_tol

    while !target_reached && iterations_done < MAX_ITERATIONS_LIMIT
        iterations_done += 1

        vals = [objective_at_gene(g, config, encoding) for g in population]
        elite_indices = partialsortperm(vals, 1:ELITISM_COUNT)

        new_population = Tuple{Int, Int}[]
        for idx in elite_indices
            push!(new_population, population[idx])
        end

        while length(new_population) < POPULATION_SIZE
            parent1 = tournament_select(population, config, encoding, rng)
            parent2 = tournament_select(population, config, encoding, rng)

            child1, child2 = if rand(rng) < CROSSOVER_RATE
                one_point_crossover_2d(parent1, parent2, crossover_point, encoding)
            else
                (parent1, parent2)
            end

            child1 = mutate_gene(child1, mutation_rate, encoding, rng)
            child2 = mutate_gene(child2, mutation_rate, encoding, rng)

            push!(new_population, child1)
            if length(new_population) < POPULATION_SIZE
                push!(new_population, child2)
            end
        end

        population = new_population
        save_generation!(population)
        target_reached = best_values_history[end] <= target_value + config.target_tol
    end

    best_iter = argmin(best_values_history)
    stop_reason = target_reached ? "target_minimum_reached" : "max_iterations_limit_reached"

    return (
        best_point = best_point_history[best_iter],
        best_value = best_values_history[best_iter],
        best_iteration = best_iter - 1,
        iterations_done = iterations_done,
        target_reached = target_reached,
        stop_reason = stop_reason,
        population_history = population_history,
        best_values_history = best_values_history,
        best_point_history = best_point_history,
        mutation_rate = mutation_rate,
        crossover_point = crossover_point,
        seed = seed,
        target_value = target_value,
    )
end

function run_ga_with_restarts(
    config::FunctionConfig,
    encoding::EncodingConfig;
    mutation_rate::Float64 = DEFAULT_MUTATION_RATE,
    crossover_point::Int = default_crossover_point(encoding),
    initial_population_points::Vector{Tuple{Float64, Float64}} = build_initial_population(config),
    base_seed::Int = RNG_SEED,
    max_restarts::Int = BASE_MAX_RESTARTS,
)
    best_result = nothing

    for attempt in 0:(max_restarts - 1)
        seed = base_seed + attempt
        result = run_ga(
            config,
            encoding;
            mutation_rate = mutation_rate,
            crossover_point = crossover_point,
            initial_population_points = initial_population_points,
            seed = seed,
        )

        if isnothing(best_result) || result.best_value < best_result.best_value
            best_result = result
        end

        if result.target_reached
            return (
                result = result,
                reached_target = true,
                attempts_used = attempt + 1,
            )
        end
    end

    return (
        result = best_result,
        reached_target = false,
        attempts_used = max_restarts,
    )
end

function sample_generation_indices(total_count::Int; limit::Int = TRAJECTORY_SAMPLE_LIMIT)
    if total_count <= limit
        return collect(1:total_count)
    end

    indices = unique(round.(Int, range(1, total_count; length = limit)))
    if indices[1] != 1
        insert!(indices, 1, 1)
    end
    if indices[end] != total_count
        push!(indices, total_count)
    end
    return indices
end

safe_gap(v::Float64, target_value::Float64) = max(abs(v - target_value), CONVERGENCE_GAP_EPS)

function plot_search_process_2d(
    result,
    config::FunctionConfig;
    title_text::String = "ГА: поиск минимума в 2D",
)
    grid_x = range(config.x_min, config.x_max; length = PLOT_GRID_RESOLUTION)
    grid_y = range(config.y_min, config.y_max; length = PLOT_GRID_RESOLUTION)
    grid_z = [config.objective(x, y) for y in grid_y, x in grid_x]

    p3d = surface(
        grid_x,
        grid_y,
        grid_z;
        xlabel = "x",
        ylabel = "y",
        zlabel = "f(x, y)",
        title = "$(config.name): 3D поверхность",
        color = :viridis,
        alpha = 0.85,
        legend = false,
        right_margin = 10Plots.mm,
        bottom_margin = 6Plots.mm,
        camera = (45, 30),
    )

    p2d = contourf(
        grid_x,
        grid_y,
        grid_z;
        xlabel = "x",
        ylabel = "y",
        title = "$(config.name): вид сверху",
        levels = 36,
        color = :viridis,
        legend = :outerright,
        right_margin = 22Plots.mm,
        bottom_margin = 6Plots.mm,
    )

    init_points = result.population_history[1]
    init_x = [p[1] for p in init_points]
    init_y = [p[2] for p in init_points]
    init_z = [objective_at_point(p, config) for p in init_points]

    scatter!(
        p3d,
        init_x,
        init_y,
        init_z;
        marker = :xcross,
        ms = 4,
        mc = :blue,
        label = false,
    )

    scatter!(
        p2d,
        init_x,
        init_y;
        marker = :xcross,
        ms = 4,
        mc = :blue,
        label = "Начальная популяция",
    )

    sampled_indices = sample_generation_indices(length(result.best_point_history))
    best_path = result.best_point_history[sampled_indices]

    if length(best_path) > 1
        traj_x = [p[1] for p in best_path]
        traj_y = [p[2] for p in best_path]
        traj_z = [objective_at_point(p, config) for p in best_path]

        plot!(
            p3d,
            traj_x,
            traj_y,
            traj_z;
            color = :red,
            alpha = 0.85,
            linewidth = 2.0,
            label = false,
        )

        scatter!(
            p3d,
            traj_x,
            traj_y,
            traj_z;
            marker = :circle,
            ms = 2.0,
            color = :red,
            alpha = 0.70,
            label = false,
        )

        plot!(
            p2d,
            traj_x,
            traj_y;
            color = :red,
            alpha = 0.85,
            linewidth = 2.0,
            label = "Траектория лучшей точки",
        )

        scatter!(
            p2d,
            traj_x,
            traj_y;
            marker = :circle,
            ms = 2.0,
            color = :red,
            alpha = 0.70,
            label = false,
        )
    end

    final_points = result.population_history[end]
    final_x = [p[1] for p in final_points]
    final_y = [p[2] for p in final_points]
    final_z = [objective_at_point(p, config) for p in final_points]

    scatter!(
        p3d,
        final_x,
        final_y,
        final_z;
        marker = :circle,
        ms = 3,
        color = :black,
        label = false,
    )

    scatter!(
        p2d,
        final_x,
        final_y;
        marker = :circle,
        ms = 3,
        color = :black,
        label = "Финальная популяция",
    )

    best_x, best_y = result.best_point
    best_z = result.best_value

    scatter!(
        p3d,
        [best_x],
        [best_y],
        [best_z];
        marker = :star5,
        ms = 9,
        color = :green,
        label = false,
    )

    scatter!(
        p2d,
        [best_x],
        [best_y];
        marker = :star5,
        ms = 9,
        color = :green,
        label = "Лучшее найденное",
    )

    target_x, target_y = config.target_min_point
    target_z = target_min_value(config)

    scatter!(
        p3d,
        [target_x],
        [target_y],
        [target_z];
        marker = :diamond,
        ms = 7,
        color = :white,
        markerstrokecolor = :black,
        label = false,
    )

    scatter!(
        p2d,
        [target_x],
        [target_y];
        marker = :diamond,
        ms = 7,
        color = :white,
        markerstrokecolor = :black,
        label = "Известный минимум",
    )

    return plot(
        p3d,
        p2d;
        layout = (1, 2),
        size = (1700, 700),
        plot_title = title_text,
        plot_titlefontsize = 13,
    )
end

function plot_convergence(
    results::Vector{Tuple{String, Any}},
    target_value::Float64;
    title_text::String,
)
    p = plot(
        title = title_text,
        xlabel = "Итерация",
        ylabel = "|f_best - f*| (log10)",
        legend = :outerright,
        linewidth = 2,
        yscale = :log10,
        size = (1250, 700),
        right_margin = 22Plots.mm,
        bottom_margin = 6Plots.mm,
    )

    for (label, result) in results
        iterations = 1:length(result.best_values_history)
        gaps = [safe_gap(v, target_value) for v in result.best_values_history]
        plot!(p, iterations, gaps; label = label)
    end

    return p
end

function get_crossover_points_to_test(encoding::EncodingConfig)
    max_point = max_valid_crossover_point(encoding)
    @assert max_point >= 1 "Недостаточно бит для теста точек кроссинговера"

    raw_points = [Int(round(r * max_point)) for r in CROSSOVER_POINT_RATIOS]
    clipped = [clamp(p, 1, max_point) for p in raw_points]
    return unique(clipped)
end

function run_mutation_experiments(
    config::FunctionConfig,
    encoding::EncodingConfig,
    crossover_point::Int,
    initial_population_points::Vector{Tuple{Float64, Float64}},
)
    results = Tuple{String, Any}[]
    for (i, pm) in enumerate(MUTATION_RATES_TO_TEST)
        result = run_ga(
            config,
            encoding;
            mutation_rate = pm,
            crossover_point = crossover_point,
            initial_population_points = initial_population_points,
            seed = RNG_SEED + i,
        )
        label = "pm = $(round(pm * 100; digits = 1))%"
        push!(results, (label, result))
    end
    return results
end

function run_crossover_experiments(
    config::FunctionConfig,
    encoding::EncodingConfig,
    mutation_rate::Float64,
    initial_population_points::Vector{Tuple{Float64, Float64}},
)
    results = Tuple{String, Any}[]
    points = get_crossover_points_to_test(encoding)

    for (i, cp) in enumerate(points)
        result = run_ga(
            config,
            encoding;
            mutation_rate = mutation_rate,
            crossover_point = cp,
            initial_population_points = initial_population_points,
            seed = RNG_SEED + 100 + i,
        )
        label = "cp = $cp"
        push!(results, (label, result))
    end

    return results
end

function run_function_pipeline(config::FunctionConfig)
    encoding = build_encoding(config)
    target_value = target_min_value(config)
    default_cp = default_crossover_point(encoding)
    initial_population_points = build_initial_population(config)

    println("\n", "="^88)
    println("Функция: $(config.name)")
    println("Диапазон x: [$(config.x_min), $(config.x_max)]")
    println("Диапазон y: [$(config.y_min), $(config.y_max)]")
    println("Шаг дискретизации: $GRID_STEP")
    println("Биты по x: $(encoding.x_bits), уровни: $(encoding.x_levels)")
    println("Биты по y: $(encoding.y_bits), уровни: $(encoding.y_levels)")
    println("Известный минимум: point=$(config.target_min_point), f*=$(round(target_value; digits = 8))")
    println("Допуск достижения минимума: $(config.target_tol)")
    println("Размер популяции: $POPULATION_SIZE")
    println("Максимум итераций: $MAX_ITERATIONS_LIMIT")
    println("Перезапуски базового прогона: до $BASE_MAX_RESTARTS")

    base_run = run_ga_with_restarts(
        config,
        encoding;
        mutation_rate = DEFAULT_MUTATION_RATE,
        crossover_point = default_cp,
        initial_population_points = initial_population_points,
        base_seed = RNG_SEED,
        max_restarts = BASE_MAX_RESTARTS,
    )

    base_result = base_run.result

    println("\nБазовый запуск ГА:")
    println(" mutation_rate = $(DEFAULT_MUTATION_RATE)")
    println(" crossover_point = $(default_cp)")
    println(" best_point = ($(round(base_result.best_point[1]; digits = 6)), $(round(base_result.best_point[2]; digits = 6)))")
    println(" best_value = $(round(base_result.best_value; digits = 8))")
    println(" best_iteration = $(base_result.best_iteration)")
    println(" iterations_done = $(base_result.iterations_done)")
    println(" stop_reason = $(base_result.stop_reason)")
    println(" restart_attempts_used = $(base_run.attempts_used)")
    println(" reached_target = $(base_run.reached_target)")

    p_search = plot_search_process_2d(
        base_result,
        config;
        title_text = "$(config.name): ГА, 3D поверхность и 2D проекция сверху",
    )
    search_file = joinpath("lab17", "ga2d_$(config.slug)_search_process.png")
    savefig(p_search, search_file)
    display(p_search)

    mutation_results = run_mutation_experiments(
        config,
        encoding,
        default_cp,
        initial_population_points,
    )

    p_mut = plot_convergence(
        mutation_results,
        target_value;
        title_text = "$(config.name): сходимость при разных процентах мутации",
    )
    mut_file = joinpath("lab17", "ga2d_$(config.slug)_convergence_mutation.png")
    savefig(p_mut, mut_file)
    display(p_mut)

    println("\nРезультаты для разных процентов мутации:")
    for (label, result) in mutation_results
        println(
            " $label -> best_point=($(round(result.best_point[1]; digits = 6)), $(round(result.best_point[2]; digits = 6))), " *
            "best_f=$(round(result.best_value; digits = 8)), iter=$(result.iterations_done), stop=$(result.stop_reason)"
        )
    end

    crossover_results = run_crossover_experiments(
        config,
        encoding,
        FIXED_MUTATION_RATE,
        initial_population_points,
    )

    p_cross = plot_convergence(
        crossover_results,
        target_value;
        title_text = "$(config.name): сходимость при разных точках кроссинговера",
    )
    cross_file = joinpath("lab17", "ga2d_$(config.slug)_convergence_crossover.png")
    savefig(p_cross, cross_file)
    display(p_cross)

    println("\nРезультаты для разных точек кроссинговера (mutation_rate=$(FIXED_MUTATION_RATE)):")
    for (label, result) in crossover_results
        println(
            " $label -> best_point=($(round(result.best_point[1]; digits = 6)), $(round(result.best_point[2]; digits = 6))), " *
            "best_f=$(round(result.best_value; digits = 8)), iter=$(result.iterations_done), stop=$(result.stop_reason)"
        )
    end

    println("\nСохраненные графики для $(config.name):")
    println(" - $(search_file)")
    println(" - $(mut_file)")
    println(" - $(cross_file)")
end

function main()
    println("Запуск двумерного генетического алгоритма для трех benchmark-функций")
    println("Mutation rates: $(MUTATION_RATES_TO_TEST)")
    println("Crossover point ratios: $(CROSSOVER_POINT_RATIOS)")
    println("Seed: $(RNG_SEED)")

    for config in FUNCTION_CONFIGS
        run_function_pipeline(config)
    end

    println("\nВсе функции обработаны.")
end

main()



