

ENV["GKSwstype"] = "100"

using Random
using Statistics
using Printf
using FileIO
using ImageIO
using Colors
using Plots
plotlyjs()

Base.@kwdef struct GAParams
    population_size::Int = 28
    max_generations::Int = 1000
    elitism_count::Int = 3
    tournament_size::Int = 3
    crossover_rate::Float64 = 0.95
    mutation_rate_start::Float64 = 0.14
    mutation_rate_end::Float64 = 0.01
    mutation_sigma_start::Float32 = 0.10f0
    mutation_sigma_end::Float32 = 0.01f0
    guided_pull_start::Float32 = 0.20f0
    guided_pull_end::Float32 = 0.75f0
    init_noise_sigma::Float32 = 0.25f0
    mse_threshold::Float64 = 0.00001
    seed::Int = 42
    fps::Int = 100
    export_mp4::Bool = true
    frame_stride::Int = 1
    block_size_min_ratio::Float32 = 0.08f0
    block_size_max_ratio::Float32 = 0.35f0
    stagnation_generations::Int = 60
    immigrants_count::Int = 2
    immigrant_noise_sigma::Float32 = 0.15f0
end

function image_to_tensor(image)
    if ndims(image) == 2 && eltype(image) <: Colorant
        h, w = size(image)
        tensor = Array{Float32}(undef, h, w, 3)
        @inbounds for i in 1:h
            for j in 1:w
                pixel = RGB{Float32}(image[i, j])
                tensor[i, j, 1] = red(pixel)
                tensor[i, j, 2] = green(pixel)
                tensor[i, j, 3] = blue(pixel)
            end
        end
        return tensor
    end

    if ndims(image) == 3 && size(image, 3) == 3
        return clamp.(Float32.(image), 0f0, 1f0)
    end

    error("Unsupported image format. Expected RGB matrix.")
end

function tensor_to_rgb(tensor::Array{Float32,3})
    h, w, c = size(tensor)
    @assert c == 3 "Tensor must have exactly 3 channels"

    out = Matrix{RGB{Float32}}(undef, h, w)
    @inbounds for i in 1:h
        for j in 1:w
            out[i, j] = RGB{Float32}(tensor[i, j, 1], tensor[i, j, 2], tensor[i, j, 3])
        end
    end

    return out
end

function mse(candidate::Array{Float32,3}, target::Array{Float32,3})
    @assert size(candidate) == size(target) "candidate and target sizes must match"

    total = 0.0
    @inbounds @simd for idx in eachindex(candidate)
        d = Float64(candidate[idx] - target[idx])
        total += d * d
    end

    return total / length(candidate)
end

function initialize_population(
    source::Array{Float32,3},
    target::Array{Float32,3},
    params::GAParams,
    rng::AbstractRNG,
)
    pop = Vector{Array{Float32,3}}(undef, params.population_size)
    pop[1] = copy(source)

    for idx in 2:params.population_size
        if idx == 2
            alpha = 0.15f0
            base = (1f0 - alpha) .* source .+ alpha .* target
            individual = base .+ params.init_noise_sigma .* randn(rng, Float32, size(source))
        elseif isodd(idx)
            alpha = rand(rng, Float32) * 0.35f0
            base = (1f0 - alpha) .* source .+ alpha .* target
            individual = base .+ params.init_noise_sigma .* randn(rng, Float32, size(source))
        else
            individual = source .+ params.init_noise_sigma .* randn(rng, Float32, size(source))
        end

        individual .= clamp.(individual, 0f0, 1f0)
        pop[idx] = individual
    end

    return pop
end

function evaluate_population(population::Vector{Array{Float32,3}}, target::Array{Float32,3})
    values = Vector{Float64}(undef, length(population))
    for i in eachindex(population)
        values[i] = mse(population[i], target)
    end
    return values
end

function tournament_select_index(fitness::Vector{Float64}, params::GAParams, rng::AbstractRNG)
    best_idx = rand(rng, 1:length(fitness))
    best_val = fitness[best_idx]

    for _ in 2:params.tournament_size
        idx = rand(rng, 1:length(fitness))
        if fitness[idx] < best_val
            best_idx = idx
            best_val = fitness[idx]
        end
    end

    return best_idx
end

@inline function linear_schedule(start_value::T, end_value::T, generation::Int, max_generations::Int) where {T <: Real}
    if max_generations <= 1
        return end_value
    end
    t = (generation - 1) / (max_generations - 1)
    return start_value + (end_value - start_value) * t
end

function random_block_bounds(height::Int, width::Int, params::GAParams, rng::AbstractRNG)
    min_ratio = min(params.block_size_min_ratio, params.block_size_max_ratio)
    max_ratio = max(params.block_size_min_ratio, params.block_size_max_ratio)

    h_ratio = rand(rng, Float32) * (max_ratio - min_ratio) + min_ratio
    w_ratio = rand(rng, Float32) * (max_ratio - min_ratio) + min_ratio

    block_h = clamp(round(Int, height * h_ratio), 1, height)
    block_w = clamp(round(Int, width * w_ratio), 1, width)

    y1 = rand(rng, 1:(height - block_h + 1))
    x1 = rand(rng, 1:(width - block_w + 1))

    return y1, y1 + block_h - 1, x1, x1 + block_w - 1
end

function block_crossover(
    parent1::Array{Float32,3},
    parent2::Array{Float32,3},
    params::GAParams,
    rng::AbstractRNG,
)
    child1 = copy(parent1)
    child2 = copy(parent2)

    height, width, _ = size(parent1)
    y1, y2, x1, x2 = random_block_bounds(height, width, params, rng)

    @views begin
        child1[y1:y2, x1:x2, :] .= parent2[y1:y2, x1:x2, :]
        child2[y1:y2, x1:x2, :] .= parent1[y1:y2, x1:x2, :]
    end

    return child1, child2
end

function mutate_towards_target!(
    individual::Array{Float32,3},
    target::Array{Float32,3},
    mutation_rate::Float64,
    mutation_sigma::Float32,
    guided_pull::Float32,
    rng::AbstractRNG,
)
    @inbounds for idx in eachindex(individual)
        if rand(rng) < mutation_rate
            base = individual[idx]
            guided_value = base + guided_pull * (target[idx] - base)
            individual[idx] = clamp(guided_value + mutation_sigma * randn(rng, Float32), 0f0, 1f0)
        end
    end

    return individual
end

function make_immigrant(
    best_image::Array{Float32,3},
    target::Array{Float32,3},
    params::GAParams,
    rng::AbstractRNG,
)
    alpha = rand(rng, Float32) * 0.60f0 + 0.20f0
    immigrant = (1f0 - alpha) .* best_image .+ alpha .* target
    immigrant .+= params.immigrant_noise_sigma .* randn(rng, Float32, size(best_image))
    immigrant .= clamp.(immigrant, 0f0, 1f0)

    return immigrant
end

function inject_immigrants!(
    population::Vector{Array{Float32,3}},
    fitness::Vector{Float64},
    best_image::Array{Float32,3},
    target::Array{Float32,3},
    params::GAParams,
    rng::AbstractRNG,
)
    k = min(params.immigrants_count, length(population) - 1)
    if k <= 0
        return
    end

    worst_range = (length(fitness) - k + 1):length(fitness)
    worst_indices = partialsortperm(fitness, worst_range; rev = true)

    for idx in worst_indices
        population[idx] = make_immigrant(best_image, target, params, rng)
    end
end

function add_frame!(
    anim::Animation,
    best_image::Array{Float32,3},
    target_rgb::Matrix{RGB{Float32}},
    generation::Int,
    best_mse::Float64,
)
    best_rgb = tensor_to_rgb(best_image)

    p_best = plot(
        best_rgb,
        axis = false,
        ticks = false,
        border = :none,
        title = @sprintf("Best image | generation %d | MSE %.6f", generation, best_mse),
    )

    p_target = plot(
        target_rgb,
        axis = false,
        ticks = false,
        border = :none,
        title = "Target image",
    )

    merged = plot(p_best, p_target, layout = (1, 2), size = (300, 150), background_color = :white)
    frame(anim, merged)
end

function save_animation_outputs(anim::Animation, output_gif_path::String, params::GAParams)
    gif(anim, output_gif_path, fps = params.fps)

    output_mp4_path = ""
    if params.export_mp4
        output_mp4_path = replace(output_gif_path, r"\.gif$" => ".mp4")
        try
            mp4(anim, output_mp4_path, fps = params.fps)
        catch err
            @warn "Failed to export MP4 animation" err
            output_mp4_path = ""
        end
    end

    return output_mp4_path
end

function run_ga_with_animation(
    source::Array{Float32,3},
    target::Array{Float32,3},
    target_rgb::Matrix{RGB{Float32}},
    params::GAParams,
    output_gif_path::String,
)
    @assert params.population_size >= 2 "population_size must be >= 2"
    @assert 1 <= params.elitism_count < params.population_size "elitism_count must be in [1, population_size - 1]"
    @assert params.tournament_size >= 2 "tournament_size must be >= 2"
    @assert params.immigrants_count < params.population_size "immigrants_count must be < population_size"
    @assert params.frame_stride >= 1 "frame_stride must be >= 1"
    @assert 0.0 <= params.mutation_rate_end <= params.mutation_rate_start <= 1.0 "mutation rates must satisfy 0 <= end <= start <= 1"

    rng = MersenneTwister(params.seed)

    population = initialize_population(source, target, params, rng)
    fitness = evaluate_population(population, target)

    best_idx = argmin(fitness)
    best_image = copy(population[best_idx])
    best_mse = fitness[best_idx]

    best_mse_history = Float64[best_mse]
    generations_done = 0
    stop_reason = "max_generations_reached"
    no_improvement_generations = 0

    anim = Animation()
    frame_count = 0
    add_frame!(anim, best_image, target_rgb, 0, best_mse)
    frame_count += 1

    if best_mse <= params.mse_threshold
        stop_reason = "threshold_reached"
        output_mp4_path = save_animation_outputs(anim, output_gif_path, params)
        return (
            best_image = best_image,
            best_mse = best_mse,
            best_mse_history = best_mse_history,
            generations_done = generations_done,
            stop_reason = stop_reason,
            output_mp4_path = output_mp4_path,
            frame_count = frame_count,
        )
    end

    for generation in 1:params.max_generations
        generations_done = generation

        mutation_rate = linear_schedule(
            params.mutation_rate_start,
            params.mutation_rate_end,
            generation,
            params.max_generations,
        )
        mutation_sigma = Float32(linear_schedule(
            params.mutation_sigma_start,
            params.mutation_sigma_end,
            generation,
            params.max_generations,
        ))
        guided_pull = Float32(linear_schedule(
            params.guided_pull_start,
            params.guided_pull_end,
            generation,
            params.max_generations,
        ))

        elite_indices = partialsortperm(fitness, 1:params.elitism_count)
        new_population = Vector{Array{Float32,3}}(undef, params.population_size)

        insert_pos = 1
        for elite_idx in elite_indices
            new_population[insert_pos] = copy(population[elite_idx])
            insert_pos += 1
        end

        while insert_pos <= params.population_size
            p1_idx = tournament_select_index(fitness, params, rng)
            p2_idx = tournament_select_index(fitness, params, rng)

            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            child1, child2 = if rand(rng) < params.crossover_rate
                block_crossover(parent1, parent2, params, rng)
            else
                (copy(parent1), copy(parent2))
            end

            mutate_towards_target!(child1, target, mutation_rate, mutation_sigma, guided_pull, rng)
            new_population[insert_pos] = child1
            insert_pos += 1

            if insert_pos <= params.population_size
                mutate_towards_target!(child2, target, mutation_rate, mutation_sigma, guided_pull, rng)
                new_population[insert_pos] = child2
                insert_pos += 1
            end
        end

        population = new_population
        fitness = evaluate_population(population, target)

        current_best_idx = argmin(fitness)
        current_best_mse = fitness[current_best_idx]

        if current_best_mse < best_mse
            best_mse = current_best_mse
            best_image = copy(population[current_best_idx])
            no_improvement_generations = 0
        else
            no_improvement_generations += 1
        end

        if no_improvement_generations >= params.stagnation_generations
            inject_immigrants!(population, fitness, best_image, target, params, rng)
            fitness = evaluate_population(population, target)
            current_best_idx = argmin(fitness)
            current_best_mse = fitness[current_best_idx]
            if current_best_mse < best_mse
                best_mse = current_best_mse
                best_image = copy(population[current_best_idx])
            end
            no_improvement_generations = 0
        end

        push!(best_mse_history, best_mse)

        if generation % params.frame_stride == 0 || generation == params.max_generations || best_mse <= params.mse_threshold
            add_frame!(anim, best_image, target_rgb, generation, best_mse)
            frame_count += 1
        end

        if generation == 1 || generation % 20 == 0
            println(@sprintf(
                "Generation %4d/%4d | best-so-far MSE: %.6f | pm=%.4f sigma=%.4f pull=%.3f",
                generation,
                params.max_generations,
                best_mse,
                mutation_rate,
                Float64(mutation_sigma),
                Float64(guided_pull),
            ))
        end

        if best_mse <= params.mse_threshold
            stop_reason = "threshold_reached"
            break
        end
    end

    output_mp4_path = save_animation_outputs(anim, output_gif_path, params)

    return (
        best_image = best_image,
        best_mse = best_mse,
        best_mse_history = best_mse_history,
        generations_done = generations_done,
        stop_reason = stop_reason,
        output_mp4_path = output_mp4_path,
        frame_count = frame_count,
    )
end

function save_convergence_plot(best_mse_history::Vector{Float64}, output_plot_path::String)
    generations = 0:(length(best_mse_history) - 1)

    p = plot(
        generations,
        best_mse_history,
        xscale = :log10,
        xlabel = "Generation",
        ylabel = "Best-so-far MSE",
        title = "GA convergence",
        legend = false,
        linewidth = 2,
        color = :dodgerblue,
    )

    savefig(p, output_plot_path)
end

function main()
    script_dir = "lab18"
    target_path = joinpath(script_dir, "mario.jpg")
    output_gif_path = joinpath(script_dir, "mario_progress.gif")
    output_best_path = joinpath(script_dir, "mario_best.png")
    output_curve_path = joinpath(script_dir, "mario_convergence.png")

    if !isfile(target_path)
        error("Target image not found: $(target_path)")
    end

    params = GAParams()

    target_raw = load(target_path)
    target_tensor = image_to_tensor(target_raw)
    target_rgb = tensor_to_rgb(target_tensor)
    source_tensor = zeros(Float32, size(target_tensor))

    h, w, _ = size(target_tensor)
    println("Target image size: $(w)x$(h)")
    println("Population size: $(params.population_size)")
    println("Max generations: $(params.max_generations)")
    println("MSE threshold: $(params.mse_threshold)")
    println("Frame stride: $(params.frame_stride)")
    println("Running GA...")

    result = run_ga_with_animation(
        source_tensor,
        target_tensor,
        target_rgb,
        params,
        output_gif_path,
    )

    save(output_best_path, tensor_to_rgb(result.best_image))
    save_convergence_plot(result.best_mse_history, output_curve_path)

    println("Done.")
    println(@sprintf("Generations completed: %d", result.generations_done))
    println(@sprintf("Final best MSE: %.6f", result.best_mse))
    println("Stop reason: $(result.stop_reason)")
    println("Saved best image: $(output_best_path)")
    println("Saved progress GIF: $(output_gif_path)")
    if !isempty(result.output_mp4_path)
        println("Saved progress MP4: $(result.output_mp4_path)")
    end
    println("Saved convergence plot: $(output_curve_path)")
    println("GIF frames: $(result.frame_count)")
end

main()

