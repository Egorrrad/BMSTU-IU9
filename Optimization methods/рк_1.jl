using Printf
using Plots

gr()
default(size=(1100, 520), margin=6Plots.mm, grid=true)


X_LO = 0.0
X_HI = 8.0

f1(x) = 5 - 24x + 17x^2 - (11/3)*x^3 + (1/4)*x^4
f2(x) = 5 - 24*(X_HI - x) + 17*(X_HI - x)^2 - (11/3)*(X_HI - x)^3 + (1/4)*(X_HI - x)^4

N_GRID = 2000
xgrid = collect(range(X_LO, X_HI, length=N_GRID))

f1_vals = f1.(xgrid)
f2_vals = f2.(xgrid)

function argmin_on_grid(vals)
    i = argmin(vals)
    return (xgrid[i], vals[i], i)
end

x1_star, f1_star, i1 = argmin_on_grid(f1_vals)
x2_star, f2_star, i2 = argmin_on_grid(f2_vals)
F_star = (f1_star, f2_star)

f2_at_x1 = f2_vals[i1]
f1_at_x2 = f1_vals[i2]

println("\n=== Идеальная (утопическая) точка ===")
@printf("min f1: x1* = %.6f, f1* = %.8f, f2(x1*) = %.8f\n", x1_star, f1_star, f2_at_x1)
@printf("min f2: x2* = %.6f, f2* = %.8f, f1(x2*) = %.8f\n", x2_star, f2_star, f1_at_x2)
@printf("F* (утопическая) = (f1*, f2*) = (%.8f, %.8f)\n", f1_star, f2_star)

p_f = plot(xgrid, f1_vals, label="f1(x)", xlabel="x", ylabel="f(x)", title="f1(x), f2(x) на [$(X_LO), $(X_HI)]")
plot!(p_f, xgrid, f2_vals, label="f2(x)")
scatter!(p_f, [x1_star], [f1_star], label="min f1", ms=5)
scatter!(p_f, [x2_star], [f2_star], label="min f2", ms=5)

p_f

@inline function dominates(a1, a2, b1, b2)
    return (a1 <= b1) && (a2 <= b2) && ((a1 < b1) || (a2 < b2))
end

function pareto_mask(f1v, f2v)
    n = length(f1v)
    mask = trues(n)
    for i in 1:n
        if !mask[i]
            continue
        end
        for j in 1:n
            if i == j
                continue
            end
            if dominates(f1v[j], f2v[j], f1v[i], f2v[i])
                mask[i] = false
                break
            end
        end
    end
    return mask
end

pmask = pareto_mask(f1_vals, f2_vals)
pareto_x = xgrid[pmask]
pareto_f1 = f1_vals[pmask]
pareto_f2 = f2_vals[pmask]

ord = sortperm(pareto_x)
pareto_x = pareto_x[ord]
pareto_f1 = pareto_f1[ord]
pareto_f2 = pareto_f2[ord]

gap_th = (X_HI - X_LO) / 40
pareto_f1_line = Float64[]
pareto_f2_line = Float64[]
for k in 1:length(pareto_x)
    if k > 1 && (pareto_x[k] - pareto_x[k-1]) > gap_th
        push!(pareto_f1_line, NaN)
        push!(pareto_f2_line, NaN)
    end
    push!(pareto_f1_line, pareto_f1[k])
    push!(pareto_f2_line, pareto_f2[k])
end

@printf("Pareto points (grid) = %d\n", length(pareto_x))

println("\n=== SWO (Sum of Weighted Objectives) ===")

phi_swo(x, λ1, λ2) = λ1 * f1(x) + λ2 * f2(x)

function minimize_swo_on_grid(λ1, λ2)
    φ_vals = phi_swo.(xgrid, λ1, λ2)
    i = argmin(φ_vals)
    xstar = xgrid[i]
    return (xstar, φ_vals[i], (f1_vals[i], f2_vals[i]), i)
end

λ_grid = collect(range(0.05, 0.95, length=19))
lambdas = [(λ, 1 - λ) for λ in λ_grid]

swo_x = Float64[]
swo_f1 = Float64[]
swo_f2 = Float64[]
swo_idx = Int[]

println("Решения SWO (минимизация λ1 f1 + λ2 f2 на сетке x):")
for (λ1, λ2) in lambdas
    xstar, φstar, (ff1, ff2), i = minimize_swo_on_grid(λ1, λ2)
    push!(swo_x, xstar)
    push!(swo_f1, ff1)
    push!(swo_f2, ff2)
    push!(swo_idx, i)
    @printf("λ=(%.3f, %.3f) -> x*=%.6f, f=(%.6f, %.6f), φ=%.8f\n", λ1, λ2, xstar, ff1, ff2, φstar)
end


λ_show = [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)]

p_phi = plot(xlabel="x", ylabel="φ(x)", title="Взвешенная сумма φ(x) для разных λ")
for (λ1, λ2) in λ_show
    φ_vals = phi_swo.(xgrid, λ1, λ2)
    i = argmin(φ_vals)
    plot!(p_phi, xgrid, φ_vals, label=@sprintf("λ=(%.1f,%.1f)", λ1, λ2))
    scatter!(p_phi, [xgrid[i]], [φ_vals[i]], ms=4, label="")
end

p_phi

function check_theorem_swo_grid(xstar_idx)
    f1s = f1_vals[xstar_idx]
    f2s = f2_vals[xstar_idx]
    for j in eachindex(xgrid)
        if dominates(f1_vals[j], f2_vals[j], f1s, f2s)
            return (false, j)
        end
    end
    return (true, 0)
end

println("\nПроверка теоремы 1 (недоминируемость найденных x* на сетке):")
all_ok = true
for (k, (λ1, λ2)) in enumerate(lambdas)
    i = swo_idx[k]
    ok, j = check_theorem_swo_grid(i)
    all_ok &= ok
    if ok
        @printf("k=%02d λ=(%.3f,%.3f): OK (x* недоминируем на сетке)\n", k, λ1, λ2)
    else
        @printf("k=%02d λ=(%.3f,%.3f): FAIL (доминирует x=%.6f, f=(%.6f,%.6f))\n",
            k, λ1, λ2, xgrid[j], f1_vals[j], f2_vals[j])
    end
end
println(all_ok ? "Итог: на выбранной сетке теорема подтверждается для всех весов." :
                "Итог: найдены нарушения на сетке (возможна грубая сетка/численная погрешность).")


p = scatter(f1_vals, f2_vals, ms=2, alpha=0.20, label="D_F (сеточно)",
    xlabel="f1", ylabel="f2", title="SWO: решения, фронт Парето и идеальная точка")
plot!(p, pareto_f1_line, pareto_f2_line, lw=2, label="Фронт Парето (по x)")

scatter!(p, swo_f1, swo_f2, ms=5, label="SWO решения", markerstrokewidth=0)

scatter!(p, [f1_star], [f2_at_x1], ms=7, marker=:diamond, label="x1*: min f1 (достижимая точка)")
scatter!(p, [f1_at_x2], [f2_star], ms=7, marker=:diamond, label="x2*: min f2 (достижимая точка)")

scatter!(p, [F_star[1]], [F_star[2]], ms=9, marker=:star5, label="F* (утопическая)")

p
