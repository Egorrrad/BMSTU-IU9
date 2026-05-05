using Random
using Statistics
using LinearAlgebra
using Plots

plotlyjs()
default(fmt = :html)

function rosenbrock(x)
    return 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2
end

function rastrigin(x)
    return 20 + x[1]^2 + x[2]^2 - 10 * (cos(2 * x[1]) + cos(2 * x[2]))
end

function schwefel(x)
    d = length(x)
    return 418.9829 * d - sum(xi * sin(sqrt(abs(xi))) for xi in x)
end

clamp_vec(x, lb, ub) = clamp.(x, lb, ub)

function rand_in_bounds(dim, lb, ub; rng=Random.default_rng())
    return lb .+ rand(rng, dim) .* (ub .- lb)
end

function safe_mean(points)
    pts = collect(points)
    isempty(pts) && error("safe_mean")
    dim = length(first(pts))
    c = zeros(Float64, dim)
    for p in pts
        c .+= p
    end
    return c ./ length(pts)
end

function detect_stagnation(best_history; m=20, eta=1e-4)
    if length(best_history) <= m
        return false
    end
    return abs(best_history[end - m] - best_history[end]) < eta
end

function pso(f, dim;
    num_particles=20,
    max_iter=500,
    w=0.8,
    c1=1.0,
    c2=1.0,
    bounds=(-50.0, 50.0),
    stop_window=50,
    stop_tol=1e-8,
    rng=Random.default_rng()
)
    lb_s, ub_s = bounds
    lb = fill(lb_s, dim)
    ub = fill(ub_s, dim)

    positions = [rand_in_bounds(dim, lb, ub) for _ in 1:num_particles]
    velocities = [rand(rng, dim) .* (ub .- lb) ./ 4 for _ in 1:num_particles]

    p_best = copy.(positions)
    p_best_values = [f(pos) for pos in positions]

    g_best_idx = argmin(p_best_values)
    g_best = copy(p_best[g_best_idx])
    g_best_value = p_best_values[g_best_idx]

    trajectories = [copy.(positions)]
    values = [g_best_value]
    iters = 0

    vmax = 0.2 .* (ub .- lb)

    for iter in 1:max_iter
        for i in 1:num_particles
            r1 = rand(rng, dim)
            r2 = rand(rng, dim)

            velocities[i] = w .* velocities[i] .+
                            c1 .* r1 .* (p_best[i] .- positions[i]) .+
                            c2 .* r2 .* (g_best .- positions[i])

            velocities[i] = clamp.(velocities[i], .-vmax, vmax)
            positions[i] .+= velocities[i]
            positions[i] = clamp_vec(positions[i], lb, ub)

            val = f(positions[i])
            if val < p_best_values[i]
                p_best[i] = copy(positions[i])
                p_best_values[i] = val
            end
        end

        new_val, idx = findmin(p_best_values)
        if new_val < g_best_value
            g_best = copy(p_best[idx])
            g_best_value = new_val
        end

        push!(trajectories, copy.(positions))
        push!(values, g_best_value)
        iters = iter

        if length(values) > stop_window && std(values[end-stop_window:end]) < stop_tol
            break
        end
    end

    return (
        best_position = g_best,
        best_value = g_best_value,
        iterations = iters,
        trajectories = trajectories,
        history = values,
    )
end

function random_move(x, lb, ub, visual; rng=Random.default_rng())
    dim = length(x)
    y = similar(x)
    for k in 1:dim
        if rand(rng) < 0.5
            α = rand(rng)
            y[k] = x[k] + α * min(visual, ub[k] - x[k])
        else
            α = rand(rng)
            y[k] = x[k] - α * min(visual, x[k] - lb[k])
        end
    end
    return clamp_vec(y, lb, ub)
end

function leap_move(x, lb, ub; rng=Random.default_rng())
    dim = length(x)
    y = copy(x)
    for k in 1:dim
        r = rand(rng)
        α = rand(rng)
        if r < 0.5
            y[k] = x[k] + α * (ub[k] - x[k])
        else
            y[k] = x[k] - α * (x[k] - lb[k])
        end
    end
    return clamp_vec(y, lb, ub)
end

function afsa(f, dim;
    num_fish=20,
    max_iter=500,
    bounds=(-50.0, 50.0),
    visual=nothing,
    theta=1.0,
    stagnation_m=nothing,
    stagnation_eta=1e-4,
    rng=Random.default_rng()
)
    lb_s, ub_s = bounds
    lb = fill(lb_s, dim)
    ub = fill(ub_s, dim)

    visual_val = isnothing(visual) ? 0.15 * (ub_s - lb_s) : visual
    m = isnothing(stagnation_m) ? 10 * dim : stagnation_m

    positions = [rand_in_bounds(dim, lb, ub) for _ in 1:num_fish]
    values = [f(x) for x in positions]

    best_idx = argmin(values)
    g_best = copy(positions[best_idx])
    g_best_value = values[best_idx]

    trajectories = [copy.(positions)]
    history = [g_best_value]
    iters = 0

    for iter in 1:max_iter
        new_positions = copy.(positions)
        new_values = copy(values)

        for i in 1:num_fish
            xi = positions[i]
            fi = values[i]

            neighbors = Int[]
            for j in 1:num_fish
                if j != i && norm(positions[j] .- xi) <= visual_val
                    push!(neighbors, j)
                end
            end

            yi = copy(xi)

            if isempty(neighbors)
                yi = random_move(xi, lb, ub, visual_val; rng=rng)
            else
                filled = (length(neighbors) / num_fish) > theta

                if filled
                    j = rand(rng, neighbors)
                    r = rand(rng)
                    yi = xi .+ r .* (positions[j] .- xi)
                    yi = clamp_vec(yi, lb, ub)
                else
                    centroid = safe_mean(positions[j] for j in neighbors)
                    if f(centroid) < fi
                        r = rand(rng)
                        yi = xi .+ r .* (centroid .- xi)
                        yi = clamp_vec(yi, lb, ub)
                    else
                        j = rand(rng, neighbors)
                        r = rand(rng)
                        yi = xi .+ r .* (positions[j] .- xi)
                        yi = clamp_vec(yi, lb, ub)
                    end

                    neigh_vals = [values[j] for j in neighbors]
                    local_best_j = neighbors[argmin(neigh_vals)]
                    if values[local_best_j] < fi
                        r = rand(rng)
                        yi2 = xi .+ r .* (positions[local_best_j] .- xi)
                        yi2 = clamp_vec(yi2, lb, ub)
                        if f(yi2) < f(yi)
                            yi = yi2
                        end
                    else
                        j = rand(rng, neighbors)
                        r = rand(rng)
                        yi2 = xi .+ r .* (positions[j] .- xi)
                        yi2 = clamp_vec(yi2, lb, ub)
                        if f(yi2) < f(yi)
                            yi = yi2
                        end
                    end
                end
            end

            fyi = f(yi)
            if fyi < fi
                new_positions[i] = yi
                new_values[i] = fyi
            else
                new_positions[i] = xi
                new_values[i] = fi
            end
        end

        positions = new_positions
        values = new_values

        best_idx = argmin(values)
        if values[best_idx] < g_best_value
            g_best = copy(positions[best_idx])
            g_best_value = values[best_idx]
        end

        push!(history, g_best_value)
        if detect_stagnation(history; m=m, eta=stagnation_eta)
            j = rand(rng, 1:num_fish)
            positions[j] = leap_move(positions[j], lb, ub; rng=rng)
            values[j] = f(positions[j])
            if values[j] < g_best_value
                g_best = copy(positions[j])
                g_best_value = values[j]
                history[end] = g_best_value
            end
        end

        push!(trajectories, copy.(positions))
        iters = iter
    end

    return (
        best_position = g_best,
        best_value = g_best_value,
        iterations = iters,
        trajectories = trajectories,
        history = history,
    )
end

function pad_histories(histories)
    maxlen = maximum(length.(histories))
    padded = zeros(Float64, length(histories), maxlen)
    for (i, h) in enumerate(histories)
        padded[i, 1:length(h)] .= h
        if length(h) < maxlen
            padded[i, length(h)+1:end] .= h[end]
        end
    end
    return padded
end

function run_experiment(alg, f, dim; nruns=20, seed=42, kwargs...)
    histories = Vector{Vector{Float64}}()
    finals = Float64[]
    best_points = Vector{Vector{Float64}}()

    for run in 1:nruns
        rng = MersenneTwister(seed + run)
        res = alg(f, dim; rng=rng, kwargs...)
        push!(histories, res.history)
        push!(finals, res.best_value)
        push!(best_points, res.best_position)
    end

    padded = pad_histories(histories)
    mean_curve = vec(mean(padded, dims=1))
    median_curve = vec(mapslices(median, padded; dims=1))
    q25_curve = vec(mapslices(x -> quantile(x, 0.25), padded; dims=1))
    q75_curve = vec(mapslices(x -> quantile(x, 0.75), padded; dims=1))

    return (
        histories = histories,
        finals = finals,
        mean_curve = mean_curve,
        median_curve = median_curve,
        q25_curve = q25_curve,
        q75_curve = q75_curve,
        best_points = best_points,
    )
end

function plot_comparison_curve(name, pso_stats, afsa_stats; outpath)
    x1 = 0:length(pso_stats.mean_curve)-1
    x2 = 0:length(afsa_stats.mean_curve)-1

    p = plot(
        title = "Сравнение PSO и AFSA: $(name)",
        xlabel = "Итерация",
        ylabel = "Лучшее найденное значение",
        legend = :topright,
        linewidth = 2,
        size = (1000, 600)
    )

    plot!(p, x1, pso_stats.mean_curve,
        ribbon=(pso_stats.mean_curve .- pso_stats.q25_curve,
                pso_stats.q75_curve .- pso_stats.mean_curve),
        label="PSO: среднее ± IQR")

    plot!(p, x2, afsa_stats.mean_curve,
        ribbon=(afsa_stats.mean_curve .- afsa_stats.q25_curve,
                afsa_stats.q75_curve .- afsa_stats.mean_curve),
        label="AFSA: среднее ± IQR")

    display(p)
    savefig(p, outpath)
    return p
end

function plot_trajectories_2d(f, pso_res, afsa_res; bounds=nothing, title_suffix="", outpath="trajectories.html")
    if isnothing(bounds)
        all_points = Float64[]

        for swarm in pso_res.trajectories
            for p in swarm
                append!(all_points, p)
            end
        end
        for swarm in afsa_res.trajectories
            for p in swarm
                append!(all_points, p)
            end
        end

        minv = minimum(all_points)
        maxv = maximum(all_points)

        margin = 0.1 * (maxv - minv)
        lb = minv - margin
        ub = maxv + margin
    else
        lb, ub = bounds
    end

    xs = range(lb, ub, length=250)
    ys = range(lb, ub, length=250)
    Z = [f([x, y]) for y in ys, x in xs]

    p1 = contour(xs, ys, Z,
        fill=true,
        levels=35,
        title="PSO $(title_suffix)",
        xlabel="x1",
        ylabel="x2",
        colorbar=true)

    for j in 1:length(pso_res.trajectories[1])
        path_x = [swarm[j][1] for swarm in pso_res.trajectories]
        path_y = [swarm[j][2] for swarm in pso_res.trajectories]
        plot!(p1, path_x, path_y, linewidth=1, label=false)
    end

    scatter!(p1,
        [pso_res.best_position[1]],
        [pso_res.best_position[2]],
        markershape=:star5,
        markersize=9,
        label=false)

    p2 = contour(xs, ys, Z,
        fill=true,
        levels=35,
        title="AFSA $(title_suffix)",
        xlabel="x1",
        ylabel="x2",
        colorbar=true)

    for j in 1:length(afsa_res.trajectories[1])
        path_x = [swarm[j][1] for swarm in afsa_res.trajectories]
        path_y = [swarm[j][2] for swarm in afsa_res.trajectories]
        plot!(p2, path_x, path_y, linewidth=1, label=false)
    end

    scatter!(p2,
        [afsa_res.best_position[1]],
        [afsa_res.best_position[2]],
        markershape=:star5,
        markersize=9,
        label=false)

    p = plot(p1, p2, layout=(1, 2), size=(1200, 500))
    display(p)
    savefig(p, outpath)
    return p
end

function main()
    mkpath("let6")

    test_set = [
        ("Rosenbrock", rosenbrock, 2, (-5.0, 5.0)),
        ("Rastrigin", rastrigin, 2, (-5.12, 5.12)),
        ("Schwefel", schwefel, 2, (-500.0, 500.0)),
    ]

    println("=== PSO и AFSA ===")

    for (name, f, dim, bounds) in test_set
        println("\n--- $(name) ---")

        pso_stats = run_experiment(
            pso, f, dim;
            nruns=20,
            seed=100,
            num_particles=20,
            max_iter=500,
            w=0.8,
            c1=1.0,
            c2=1.0,
            bounds=bounds,
            stop_window=50,
            stop_tol=1e-8
        )

        afsa_stats = run_experiment(
            afsa, f, dim;
            nruns=20,
            seed=100,
            num_fish=20,
            max_iter=500,
            bounds=bounds,
            visual=0.15 * (bounds[2] - bounds[1]),
            theta=0.25,
            stagnation_m=10 * dim,
            stagnation_eta=1e-4
        )

        println("PSO: median(final) = $(median(pso_stats.finals)), mean(final) = $(mean(pso_stats.finals))")
        println("AFSA: median(final) = $(median(afsa_stats.finals)), mean(final) = $(mean(afsa_stats.finals))")

        plot_comparison_curve(name, pso_stats, afsa_stats;
            outpath="let6/$(lowercase(name))_convergence.html")

        demo_pso = pso(f, dim;
            num_particles=20,
            max_iter=120,
            w=0.8,
            c1=1.0,
            c2=1.0,
            bounds=bounds,
            rng=MersenneTwister(7)
        )

        demo_afsa = afsa(f, dim;
            num_fish=20,
            max_iter=120,
            bounds=bounds,
            visual=0.15 * (bounds[2] - bounds[1]),
            theta=0.25,
            stagnation_m=10 * dim,
            stagnation_eta=1e-4,
            rng=MersenneTwister(7)
        )

        plot_trajectories_2d(f, demo_pso, demo_afsa;
            bounds=nothing,
            title_suffix="на $(name)",
            outpath="let6/$(lowercase(name))_trajectories.html")
    end

end

main()
