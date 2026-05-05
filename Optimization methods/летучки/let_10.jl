
using Random
using LinearAlgebra
using Statistics
using Printf
using Plots
plotlyjs()

const NUM_ANTS = 10
const NUM_ITERATIONS = 40
const ALPHA = 1.0
const BETA = 4.0
const RHO = 0.45
const QVAL = 100.0
const TAU0 = 1.0
const RNG_SEED = 42

const POINTS = [
    (565.0, 575.0), (25.0, 185.0), (345.0, 750.0), (945.0, 685.0),
    (845.0, 655.0), (880.0, 660.0), (25.0, 230.0), (525.0, 1000.0),
    (580.0, 1175.0), (650.0, 1130.0), (1605.0, 620.0), (1220.0, 580.0),
    (1465.0, 200.0), (1530.0, 5.0), (845.0, 680.0), (725.0, 370.0),
    (145.0, 665.0), (415.0, 635.0), (510.0, 875.0), (560.0, 365.0),
    (300.0, 465.0), (520.0, 585.0), (480.0, 415.0), (835.0, 625.0),
    (975.0, 580.0), (1215.0, 245.0), (1320.0, 315.0), (1250.0, 400.0),
    (660.0, 180.0), (410.0, 250.0), (420.0, 555.0), (575.0, 665.0),
    (1150.0, 1160.0), (700.0, 580.0), (685.0, 595.0), (685.0, 610.0),
    (770.0, 610.0), (795.0, 645.0), (720.0, 635.0), (760.0, 650.0),
    (475.0, 960.0), (95.0, 260.0), (875.0, 920.0), (700.0, 500.0),
    (555.0, 815.0), (830.0, 485.0), (1170.0, 65.0), (830.0, 610.0),
    (605.0, 625.0), (595.0, 360.0), (1340.0, 725.0), (1740.0, 245.0)
]

function euclidean(p1::Tuple{Float64,Float64}, p2::Tuple{Float64,Float64})
    sqrt((p1[1] - p2[1])^2 + (p1[2] - p2[2])^2)
end

function build_distance_matrix(points)
    n = length(points)
    D = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        if i != j
            D[i, j] = euclidean(points[i], points[j])
        end
    end
    D
end

function route_length(route::Vector{Int}, D::Matrix{Float64})
    total = 0.0
    for i in 1:length(route)-1
        total += D[route[i], route[i+1]]
    end
    total
end

function edges_of_route(route::Vector{Int})
    edges = Tuple{Int,Int}[]
    for i in 1:length(route)-1
        a, b = route[i], route[i+1]
        push!(edges, (min(a, b), max(a, b)))
    end
    edges
end

function canonical_cycle(route_closed::Vector{Int})
    route = route_closed[1:end-1]
    n = length(route)
    rots1 = [vcat(route[k:end], route[1:k-1]) for k in 1:n]
    rroute = reverse(route)
    rots2 = [vcat(rroute[k:end], rroute[1:k-1]) for k in 1:n]
    candidates = vcat(rots1, rots2)
    best = candidates[1]
    for c in candidates[2:end]
        if Tuple(c) < Tuple(best)
            best = c
        end
    end
    vcat(best, best[1])
end

function weighted_choice(indices::Vector{Int}, weights::Vector{Float64}, rng::AbstractRNG)
    s = sum(weights)
    if s <= 0
        return rand(rng, indices)
    end
    r = rand(rng) * s
    acc = 0.0
    for (idx, w) in zip(indices, weights)
        acc += w
        if r <= acc
            return idx
        end
    end
    indices[end]
end

function build_ant_route(start_node::Int, tau::Matrix{Float64}, eta::Matrix{Float64},
    alpha::Float64, beta::Float64, rng::AbstractRNG)
    n = size(tau, 1)
    unvisited = Set(1:n)
    delete!(unvisited, start_node)
    route = [start_node]
    current = start_node

    while !isempty(unvisited)
        candidates = collect(unvisited)
        desirabilities = Float64[]
        for j in candidates
            push!(desirabilities, (tau[current, j]^alpha) * (eta[current, j]^beta))
        end
        next_node = weighted_choice(candidates, desirabilities, rng)
        push!(route, next_node)
        delete!(unvisited, next_node)
        current = next_node
    end

    push!(route, start_node)
    route
end

function plot_route(points, route::Vector{Int}, length_value::Float64; title_text="")
    xs = [points[i][1] for i in route]
    ys = [points[i][2] for i in route]
    px = [p[1] for p in points]
    py = [p[2] for p in points]

    p = scatter(px, py, markersize=3, marker=:circle, legend=false,
        aspect_ratio=:equal,
        title=title_text,
        xlabel="X", ylabel="Y")

    plot!(p, xs, ys, seriestype=:path, linewidth=2)
    p
end

function ant_system_2d(points;
    num_ants::Int=NUM_ANTS,
    num_iterations::Int=NUM_ITERATIONS,
    alpha::Float64=ALPHA,
    beta::Float64=BETA,
    rho::Float64=RHO,
    qval::Float64=QVAL,
    tau0::Float64=TAU0,
    rng_seed::Int=RNG_SEED)

    rng = MersenneTwister(rng_seed)
    n = length(points)
    D = build_distance_matrix(points)

    eta = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        if i != j
            eta[i, j] = 1.0 / D[i, j]
        end
    end

    tau = fill(tau0, n, n)
    for i in 1:n
        tau[i, i] = 0.0
    end

    best_route = Int[]
    best_length = Inf
    unique_routes = Dict{Tuple{Vararg{Int}},Float64}()
    best_lengths_by_iteration = Float64[]

    for iter in 1:num_iterations
        ant_routes = Vector{Vector{Int}}()
        ant_lengths = Float64[]

        for k in 1:num_ants
            start_node = rand(rng, 1:n)
            route = build_ant_route(start_node, tau, eta, alpha, beta, rng)
            lenv = route_length(route, D)

            push!(ant_routes, route)
            push!(ant_lengths, lenv)

            if lenv < best_length
                best_length = lenv
                best_route = copy(route)
            end

            canon = canonical_cycle(route)
            unique_routes[Tuple(canon)] = lenv
        end

        push!(best_lengths_by_iteration, best_length)

        tau .*= (1.0 - rho)

        for (route, lenv) in zip(ant_routes, ant_lengths)
            deposit = qval / lenv
            for (a, b) in edges_of_route(route)
                tau[a, b] += deposit
                tau[b, a] += deposit
            end
        end
    end

    sorted_unique = sort(collect(unique_routes), by=x -> x[2])
    routes_with_lengths = [(collect(k), v) for (k, v) in sorted_unique]

    (
        best_route=best_route,
        best_length=best_length,
        unique_routes=routes_with_lengths,
        distance_matrix=D,
        best_lengths_by_iteration=best_lengths_by_iteration
    )
end

result = ant_system_2d(POINTS)

println("Лучшая длина: ", round(result.best_length, digits=4))
println("Лучший маршрут (индексы с 0): ", [x - 1 for x in result.best_route])

p_best = plot_route(POINTS, result.best_route, result.best_length,
    title_text="Лучший маршрут\nДлина = $(round(result.best_length, digits=2))")
display(p_best)


m = min(length(result.unique_routes), 12)
best_routes = result.unique_routes[1:m]

plots_arr = []
for (idx, (route, lenv)) in enumerate(best_routes)
    title = "Маршрут $idx\nДлина = $(round(lenv, digits=2))"
    p = plot_route(POINTS, route, lenv, title_text=title)
    push!(plots_arr, p)
end

gallery = plot(plots_arr..., layout=(4, 3), size=(1350, 1100))
display(gallery)


iterations = 0:length(result.best_lengths_by_iteration)-1
p_conv = plot(iterations, result.best_lengths_by_iteration,
    linewidth=2, marker=:circle, legend=false,
    xlabel="Итерация", ylabel="Лучшая длина",
    title="Сходимость алгоритма муравьиной колонии")
display(p_conv)

