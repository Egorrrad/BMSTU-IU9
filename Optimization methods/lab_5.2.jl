
using LinearAlgebra
using Printf

function golden_section_search(f, a, b; tol=1e-9, max_iter=200)
    if a >= b
        a, b = b, a
    end
    tau = (sqrt(5) - 1) / 2
    x1 = a + (1 - tau) * (b - a)
    x2 = a + tau * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    for _ in 1:max_iter
        if abs(b - a) <= tol
            break
        end
        if f1 <= f2
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1 - tau) * (b - a)
            f1 = f(x1)
        else
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + tau * (b - a)
            f2 = f(x2)
        end
    end
    return (a + b) / 2
end

# Поиск подходящего шага alpha вдоль направления p
function find_alpha(f, x, p)
    phi(alpha) = f(x + alpha * p)
    alpha_max = 1.0
    f_prev = phi(0.0)
    for _ in 1:50
        f_test = phi(alpha_max)
        if f_test > f_prev
            break
        end
        f_prev = f_test
        alpha_max *= 2.0
    end
    alpha_max = min(alpha_max, 1e6)
    return golden_section_search(phi, 0.0, alpha_max)
end

# BFGS метод
function bfgs(f, grad_f, x0; tol=1e-3, max_iter=100, debug=false)
    n = length(x0)
    x = copy(x0)
    H = Matrix{Float64}(I, n, n)
    history = [copy(x)]

    for k in 0:max_iter-1
        g = grad_f(x)
        gnorm = norm(g)

        if debug
            println("iter $k")
            @printf("x = [%.6f, %.6f]\n", x[1], x[2])
            @printf("grad = [%.6f, %.6f]\n", g[1], g[2])
            @printf("norm(grad) = %.6f\n", gnorm)
            println("H =")
            for row in eachrow(H)
                println(join([@sprintf("%.6f", v) for v in row], " "))
            end
        end

        if gnorm <= tol
            if debug
                @printf("stop: norm(grad) = %.2e <= %.2e\n", gnorm, tol)
            end
            break
        end

        p = -H * g
        if debug
            @printf("p = [%.6f, %.6f]\n", p[1], p[2])
        end

        alpha = find_alpha(f, x, p)
        if debug
            @printf("alpha = %.6f\n", alpha)
        end

        x_new = x + alpha * p
        g_new = grad_f(x_new)

        delta_x = x_new - x
        delta_g = g_new - g

        rho_denom = dot(delta_g, delta_x)
        if debug
            @printf("x_new = [%.6f, %.6f]\n", x_new[1], x_new[2])
        end

        if abs(rho_denom) > 1e-10
            H = H - (delta_x * delta_g' * H + H * delta_g * delta_x') / rho_denom +
                 (1 + (delta_g' * H * delta_g) / rho_denom) * (delta_x * delta_x') / rho_denom
        else
            H = Matrix{Float64}(I, n, n)
        end

        x = x_new
        push!(history, copy(x))
    end

    return x, history
end

# Целевая функция варианта 4: f(x,y) = 4x^2 + 2xy + 3y^2 + x - 5y + 6
function f_test(x)
    return 4*x[1]^2 + 2*x[1]*x[2] + 3*x[2]^2 + x[1] - 5*x[2] + 6
end

# Градиент варианта 4
function grad_f_test(x)
    return [8*x[1] + 2*x[2] + 1,
            2*x[1] + 6*x[2] - 5]
end

# Точное решение 
# x = -4/11 ≈ -0.363636, y = 21/22 ≈ 0.954545.
x_exact = [-4/11, 21/22]

x0 = [1.0, 1.0]
tol = 1e-3

println("BFGS: f(x,y) = 4x^2 + 2xy + 3y^2 + x - 5y + 6")
@printf("x0 = [%.6f, %.6f], tol = %.0e\n", x0[1], x0[2], tol)
@printf("exact x* = [%.6f, %.6f]\n", x_exact[1], x_exact[2])
println()

x_opt, hist = bfgs(f_test, grad_f_test, x0; tol=tol, debug=true)

println()
println("result")
@printf("x* = [%.6f, %.6f]\n", x_opt[1], x_opt[2])
@printf("f(x*) = %.10f\n", f_test(x_opt))
@printf("norm(grad) = %.2e\n", norm(grad_f_test(x_opt)))
@printf("iters = %d\n", length(hist) - 1)
@printf("error = %.2e\n", norm(x_opt - x_exact))

