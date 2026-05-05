using LinearAlgebra

function f(x)
    if length(x) != 3
        print("Вектор x должен иметь длину 3")
        exit(1)
    end

    s = 0.0
    for i in 1:4
        c_i = 2.0 * i
        inner = 0.0

        for j in 1:3
            a_ij = j / c_i
            p_ij = a_ij^2
            inner += a_ij * (x[j] - p_ij)^2
        end

        s += c_i * exp(-inner)
    end

    return -s
end


using Random


function immun(Np, s, d, K, Nc, r)
    n = 3
    a = 2
    b = 13

    k = 0
    # 1.2
    pop = [a .+ (b - a) .* rand(n) for _ in 1:Np]
    # 1.3
    valuespop = [f(pop[i]) for i in 1:Np]

    history = Vector{Vector{Vector{Float64}}}()
    push!(history, deepcopy(pop))

    while k < K
        # 2.1
        idx = sortperm(valuespop)
        pop = pop[idx]
        valuespop = valuespop[idx]

        # 2.2 равномерное клонир.
        parents = pop[1:s]
        parent_values = valuespop[1:s]

        # clones[j][c] = c-й клон j-го родителя
        clones = [[copy(parents[j]) for _ in 1:Nc] for j in 1:s]

        # 3 мутация
        for j in 1:s
            for c in 1:Nc
                for i in 1:n
                    while true
                        u = rand()
                        x_old = clones[j][c][i]

                        if u >= 0.5
                            y = x_old + rand() * (b - x_old) * r
                        else
                            y = x_old - rand() * (x_old - a) * r
                        end

                        if a <= y <= b
                            clones[j][c][i] = y
                            break
                        end
                    end
                end
            end
        end

        # 4.1
        clone_values = [[f(clones[j][c]) for c in 1:Nc] for j in 1:s]

        # 4.2
        newpop = copy(pop)
        newvaluespop = copy(valuespop)

        for j in 1:s
            best_clone_idx = argmin(clone_values[j])
            best_clone = clones[j][best_clone_idx]
            best_clone_value = clone_values[j][best_clone_idx]

            if best_clone_value < parent_values[j]
                newpop[j] = copy(best_clone)
                newvaluespop[j] = best_clone_value
            else
                newpop[j] = copy(parents[j])
                newvaluespop[j] = parent_values[j]
            end
        end

        # 5.1
        idx_new = sortperm(newvaluespop)
        newpop = newpop[idx_new]
        newvaluespop = newvaluespop[idx_new]

        for i in (Np - d + 1):Np
            newpop[i] = a .+ (b - a) .* rand(n)
        end

        # 5.2
        for i in (Np - d + 1):Np
            newvaluespop[i] = f(newpop[i])
        end

        pop = newpop
        valuespop = newvaluespop

        push!(history, deepcopy(pop))

        # 5.3
        k += 1
    end

    # 7
    idx = argmin(valuespop)
    x_best = pop[idx]
    f_best = valuespop[idx]

    return x_best, f_best, pop, valuespop, history
end


using Plots

function plot_population(pop; best_point=nothing)
    xs = [p[1] for p in pop]
    ys = [p[2] for p in pop]
    zs = [p[3] for p in pop]

    plt = scatter(
        xs, ys, zs,
        xlabel = "x1",
        ylabel = "x2",
        zlabel = "x3",
        markersize = 4,
        color = "blue"
    )

    if best_point !== nothing
        scatter!(
            [best_point[1]], [best_point[2]], [best_point[3]],
            label = "лучшая клетка",
            markersize = 8,
            markershape = :star5,
            color = "green"
        )
    end

    return plt
end


x_best, f_best, pop, valuespop, history = immun(45,  5, 3, 100, 10, 0.2)

println("Лучшее решение: ", x_best)

all_points = vcat(history...)

plot_population(all_points; best_point=x_best)
