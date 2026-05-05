using LinearAlgebra
using Plots
plotlyjs()

function roy(f; n_particles=20, n_iter=40, xrange=(0.0, 4.0), yrange=(0.0, 4.0),
             w=0.7, c1=1.5, c2=1.5, vmax=0.4)
    dim = 2
    particles = [Float64[
        rand() * (xrange[2] - xrange[1]) + xrange[1],
        rand() * (yrange[2] - yrange[1]) + yrange[1]
    ] for _ in 1:n_particles]

    velocities = [Float64[
        (2rand() - 1) * vmax,
        (2rand() - 1) * vmax
    ] for _ in 1:n_particles]

    pbest_pos = [copy(x) for x in particles]
    pbest_val = [f(x) for x in particles]

    gbest_idx = argmin(pbest_val)
    gbest_pos = copy(pbest_pos[gbest_idx])
    gbest_val = pbest_val[gbest_idx]

    swarm_history = Vector{Vector{Vector{Float64}}}()
    gbest_history = Vector{Vector{Float64}}()

    push!(swarm_history, [copy(x) for x in particles])
    push!(gbest_history, copy(gbest_pos))

    for iter in 1:n_iter
        for i in 1:n_particles
            fx = f(particles[i])
            if fx < pbest_val[i]
                pbest_val[i] = fx
                pbest_pos[i] = copy(particles[i])
            end
        end

        gbest_idx = argmin(pbest_val)
        if pbest_val[gbest_idx] < gbest_val
            gbest_val = pbest_val[gbest_idx]
            gbest_pos = copy(pbest_pos[gbest_idx])
        end

        for i in 1:n_particles
            r1 = rand(dim)
            r2 = rand(dim)

            velocities[i] = w .* velocities[i] .+
                            c1 .* r1 .* (pbest_pos[i] .- particles[i]) .+
                            c2 .* r2 .* (gbest_pos .- particles[i])

            velocities[i][1] = clamp(velocities[i][1], -vmax, vmax)
            velocities[i][2] = clamp(velocities[i][2], -vmax, vmax)

            particles[i] = particles[i] .+ velocities[i]

            particles[i][1] = clamp(particles[i][1], xrange[1], xrange[2])
            particles[i][2] = clamp(particles[i][2], yrange[1], yrange[2])
        end

        push!(swarm_history, [copy(x) for x in particles])
        push!(gbest_history, copy(gbest_pos))
    end

    return (
        best_pos = gbest_pos,
        best_val = gbest_val,
        swarm_history = swarm_history,
        gbest_history = gbest_history
    )
end

function fish(f; n_particles=20, n_iter=40, xrange=(0.0, 4.0), yrange=(0.0, 4.0),
              delta=0.25, teta=0.5, eta=1e-8, st_win=10)
    dim = 2
    lower = [xrange[1], yrange[1]]
    upper = [xrange[2], yrange[2]]
    v = delta * maximum(upper .- lower)

    particles = [Float64[
        rand() * (xrange[2] - xrange[1]) + xrange[1],
        rand() * (yrange[2] - yrange[1]) + yrange[1]
    ] for _ in 1:n_particles]

    function clamp_point!(x)
        x[1] = clamp(x[1], xrange[1], xrange[2])
        x[2] = clamp(x[2], yrange[1], yrange[2])
        x
    end

    function neighbors(i, xs)
        xi = xs[i]
        [j for j in 1:length(xs) if j != i && norm(xs[j] .- xi) <= v]
    end

    function random_behavior(x)
        y = copy(x)
        for k in 1:dim
            r = rand()
            ro = rand()
            if r < 0.5
                y[k] = x[k] + ro * min(v, upper[k] - x[k])
            else
                y[k] = x[k] - ro * min(v, x[k] - lower[k])
            end
        end
        clamp_point!(y)
    end

    function searching_behavior(i, xs, Vi)
        isempty(Vi) && return random_behavior(xs[i])
        j = rand(Vi)
        r = rand()
        y = xs[i] .+ r .* (xs[j] .- xs[i])
        clamp_point!(y)
    end

    function following_behavior(i, xs, Vi)
        jstar = Vi[argmin([f(xs[j]) for j in Vi])]
        r = rand()
        y = xs[i] .+ r .* (xs[jstar] .- xs[i])
        clamp_point!(y)
    end

    function schooling_behavior(i, xs, Vi)
        ci = zeros(dim)
        for j in Vi
            ci .+= xs[j]
        end
        ci ./= length(Vi)
        r = rand()
        y = xs[i] .+ r .* (ci .- xs[i])
        clamp_point!(y)
    end

    function jumping_behavior(x)
        y = copy(x)
        for k in 1:dim
            r = rand()
            ro = rand()
            if r < 0.5
                y[k] = x[k] + ro * (upper[k] - x[k])
            else
                y[k] = x[k] - ro * (x[k] - lower[k])
            end
        end
        clamp_point!(y)
    end

    best_vals_history = Float64[]
    best_idx = argmin([f(x) for x in particles])
    gbest_pos = copy(particles[best_idx])
    gbest_val = f(gbest_pos)

    swarm_history = Vector{Vector{Vector{Float64}}}()
    gbest_history = Vector{Vector{Float64}}()

    push!(swarm_history, [copy(x) for x in particles])
    push!(gbest_history, copy(gbest_pos))
    push!(best_vals_history, gbest_val)

    for iter in 1:n_iter
        for i in 1:n_particles
            xi = particles[i]
            fxi = f(xi)
            Vi = neighbors(i, particles)

            yi = if isempty(Vi)
                random_behavior(xi)
            elseif length(Vi) / n_particles > teta
                searching_behavior(i, particles, Vi)
            else
                ci = zeros(dim)
                for j in Vi
                    ci .+= particles[j]
                end
                ci ./= length(Vi)
                if f(ci) < fxi
                    schooling_behavior(i, particles, Vi)
                else
                    searching_behavior(i, particles, Vi)
                end
            end

            if !isempty(Vi)
                jstar = Vi[argmin([f(particles[j]) for j in Vi])]
                if f(particles[jstar]) < fxi
                    yfollow = following_behavior(i, particles, Vi)
                    if f(yfollow) < f(yi)
                        yi = yfollow
                    end
                end
            end

            if f(yi) < fxi
                particles[i] = yi
            end
        end

        vals = [f(x) for x in particles]
        best_idx = argmin(vals)
        if vals[best_idx] < gbest_val
            gbest_val = vals[best_idx]
            gbest_pos = copy(particles[best_idx])
        end

        push!(best_vals_history, gbest_val)

        if length(best_vals_history) > st_win
            old_best = best_vals_history[end - st_win]
            new_best = best_vals_history[end]
            if abs(old_best - new_best) < eta
                j = rand(1:n_particles)
                xjump = jumping_behavior(particles[j])
                if f(xjump) < f(particles[j])
                    particles[j] = xjump
                    if f(particles[j]) < gbest_val
                        gbest_val = f(particles[j])
                        gbest_pos = copy(particles[j])
                    end
                end
            end
        end

        push!(swarm_history, [copy(x) for x in particles])
        push!(gbest_history, copy(gbest_pos))
    end

    return (
        best_pos = gbest_pos,
        best_val = gbest_val,
        swarm_history = swarm_history,
        gbest_history = gbest_history
    )
end

function plot_roy(f, result; gridN=60, xrange=(0.0, 4.0), yrange=(0.0, 4.0))
    xg = range(xrange[1], xrange[2]; length=gridN)
    yg = range(yrange[1], yrange[2]; length=gridN)
    Zf = [f([x, y]) for y in yg, x in xg]

    p = Plots.surface(xg, yg, Zf,
                xlabel="x", ylabel="y", zlabel="f(x,y)",
                alpha=0.35, colorbar=false, label="f(x,y)",
                camera=(40, 28))

    for step_swarm in result.swarm_history
        xs = [pt[1] for pt in step_swarm]
        ys = [pt[2] for pt in step_swarm]
        zs = [f(pt) for pt in step_swarm]
        Plots.scatter!(p, xs, ys, zs, ms=3, alpha=0.35, color=:blue,
                       markerstrokewidth=0, label="")
    end

    gx = [pt[1] for pt in result.gbest_history]
    gy = [pt[2] for pt in result.gbest_history]
    gz = [f(pt) for pt in result.gbest_history]

    Plots.plot!(p, gx, gy, gz, lw=3, color=:red, label="траектория лучших")

    Plots.scatter!(p, [gx[end]], [gy[end]], [gz[end]],
             ms=8, color=:red, markerstrokecolor=:black,
             markerstrokewidth=1.2, label="найденный минимум")

    p
end

f2(x) = x[1]^2 + 2*x[2]^2

res_roy = roy(f2; n_particles=25, n_iter=50, xrange=(-1.0, 4.0), yrange=(-1.0, 4.0),
              w=0.7, c1=1.6, c2=1.6, vmax=0.35)
println("best_pos (roy) = ", res_roy.best_pos)
display(plot_roy(f2, res_roy; xrange=(-1.0, 4.0), yrange=(-1.0, 4.0)))

res_fish = fish(f2; n_particles=25, n_iter=50, xrange=(-1.0, 4.0), yrange=(-1.0, 4.0),
                delta=0.25, teta=0.5, eta=0.000001)
println("best_pos (fish) = ", res_fish.best_pos)
display(plot_roy(f2, res_fish; xrange=(-1.0, 4.0), yrange=(-1.0, 4.0)))
