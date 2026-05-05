
using LinearAlgebra
using Printf
using Plots
gr()

struct DFPResult
    xs::Vector{Vector{Float64}}
    fs::Vector{Float64}
    grad_norms::Vector{Float64}
    alphas::Vector{Float64}
    Hs::Vector{Matrix{Float64}}
    As::Vector{Matrix{Float64}}
    Bs::Vector{Matrix{Float64}}
    converged::Bool
    name::String
end

function vec2(x, y)
    [Float64(x), Float64(y)]
end

function armijo_line_search(f, grad, x, p; α0=1.0, c=1e-4, ρ=0.5, max_ls=40)
    α = α0
    fx = f(x)
    gx = grad(x)
    gtp = dot(gx, p)
    for _ in 1:max_ls
        xnew = x .+ α .* p
        if f(xnew) <= fx + c * α * gtp
            return α
        end
        α *= ρ
    end
    return α
end

function dfp_parts(H, s, y)
    sy = dot(s, y)
    Hy = H * y
    yHy = dot(y, Hy)
    if abs(sy) < 1e-14 || abs(yHy) < 1e-14
        return zeros(2,2), zeros(2,2)
    end
    A = (s * s') / sy
    B = (Hy * Hy') / yHy
    return A, B
end

function dfp_optimize(f, grad, x0; max_iter=300, eps=1e-8, α0=1.0, name="DFP")
    x = copy(x0)
    H = Matrix{Float64}(I, 2, 2)
    xs = [copy(x)]
    fs = [f(x)]
    grad_norms = [norm(grad(x))]
    alphas = Float64[]
    Hs = [copy(H)]
    As = Matrix{Float64}[]
    Bs = Matrix{Float64}[]
    converged = false
    for _ in 1:max_iter
        g = grad(x)
        if norm(g) < eps
            converged = true
            break
        end
        p = -H * g
        α = armijo_line_search(f, grad, x, p; α0=α0)
        xnew = x .+ α .* p
        gnew = grad(xnew)
        s = xnew - x
        y = gnew - g
        if dot(s, y) <= 1e-14
            H = Matrix{Float64}(I, 2, 2)
        else
            A, B = dfp_parts(H, s, y)
            H += A - B
            push!(As, copy(A))
            push!(Bs, copy(B))
        end
        x = xnew
        push!(xs, copy(x))
        push!(fs, f(x))
        push!(grad_norms, norm(grad(x)))
        push!(alphas, α)
        push!(Hs, copy(H))
    end
    DFPResult(xs, fs, grad_norms, alphas, Hs, As, Bs, converged, name)
end

rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
function grad_rosenbrock(x)
    [-2*(1-x[1]) - 400*x[1]*(x[2]-x[1]^2), 200*(x[2]-x[1]^2)]
end

rastrigin(x) = 20 + (x[1]^2 - 10*cos(2π*x[1])) + (x[2]^2 - 10*cos(2π*x[2]))
function grad_rastrigin(x)
    [2*x[1] + 20π*sin(2π*x[1]), 2*x[2] + 20π*sin(2π*x[2])]
end

schwefel(x) = 2*418.9829 - (x[1]*sin(sqrt(abs(x[1]))) + x[2]*sin(sqrt(abs(x[2]))))
function grad_schwefel(x)
    g = zeros(2)
    for i in 1:2
        xi = x[i]
        if abs(xi) < 1e-12
            g[i] = -sin(sqrt(abs(xi)))
        else
            u = sqrt(abs(xi))
            s = sign(xi)
            g[i] = -(sin(u) + xi*cos(u)*s/(2u))
        end
    end
    g
end

quad_f(x) = 8x[1]^2 - 6x[1]x[2] + 5x[2]^2
function grad_quad(x)
    [16x[1]-6x[2], -6x[1]+10x[2]]
end
G = [16.0 -6.0; -6.0 10.0]
Ginv = inv(G)

function exact_alpha_quad(x, p)
    g = grad_quad(x)
    -dot(g,p) / dot(p, G*p)
end

function dfp_quad(x0; max_iter=5)
    x = copy(x0)
    H = Matrix{Float64}(I, 2, 2)
    xs = [copy(x)]
    fs = [quad_f(x)]
    As = Matrix{Float64}[]
    Bs = Matrix{Float64}[]
    for _ in 1:max_iter
        g = grad_quad(x)
        if norm(g) < 1e-12
            break
        end
        p = -H * g
        α = exact_alpha_quad(x, p)
        xnew = x .+ α .* p
        gnew = grad_quad(xnew)
        s = xnew - x
        y = gnew - g
        A, B = dfp_parts(H, s, y)
        H += A - B
        push!(As, copy(A))
        push!(Bs, copy(B))
        push!(xs, copy(xnew))
        push!(fs, quad_f(xnew))
        x = xnew
    end
    DFPResult(xs, fs, zeros(length(xs)), zeros(length(xs)-1), [], As, Bs, true, "Quadratic")
end

function plot_trajectory(f, res, xrange, yrange, fname; title="")
    xg = range(xrange[1], xrange[2], 150)
    yg = range(yrange[1], yrange[2], 150)
    Z = [f([x,y]) for y in yg, x in xg]
    px = [p[1] for p in res.xs]
    py = [p[2] for p in res.xs]
    pz = [f(p) for p in res.xs]

    l = @layout [a{0.55w} b]
    p2 = contour(xg, yg, Z, levels=25, color=:viridis, title=title*" (2D)", xlabel="x", ylabel="y")
    plot!(p2, px, py, lw=3, marker=:circle, label="path", color=:red)
    scatter!(p2, [px[1]], [py[1]], color=:green, markersize=8, label="start")
    scatter!(p2, [px[end]], [py[end]], color=:blue, markersize=8, label="end")

    p3 = surface(xg, yg, Z, camera=(35,25), title=title*" (3D)", xlabel="x", ylabel="y", zlabel="f")
    plot!(p3, px, py, pz, lw=4, color=:red, label="path")
    scatter!(p3, [px[1]], [py[1]], [pz[1]], color=:green, markersize=5)
    scatter!(p3, [px[end]], [py[end]], [pz[end]], color=:blue, markersize=5)

    plot(p2, p3, layout=l, size=(1000,450), margin=5mm)
    savefig(fname)
end

mkpath("dfp_results")

r1 = dfp_optimize(rosenbrock, grad_rosenbrock, vec2(-1.2,1.0), max_iter=300, eps=1e-8, α0=1.0, name="Rosenbrock")
@printf("Rosenbrock iter=%d converged=%s x=(%.8f %.8f) f=%.2e ||g||=%.2e\n", length(r1.xs)-1, r1.converged, r1.xs[end][1], r1.xs[end][2], r1.fs[end], r1.grad_norms[end])
plot_trajectory(rosenbrock, r1, (-2,2), (-1,3), "dfp_results/rosenbrock.png", title="Rosenbrock")

r2 = dfp_optimize(rastrigin, grad_rastrigin, vec2(0.1,0.1), max_iter=300, eps=1e-8, α0=0.5, name="Rastrigin")
@printf("Rastrigin iter=%d converged=%s x=(%.8f %.8f) f=%.2e ||g||=%.2e\n", length(r2.xs)-1, r2.converged, r2.xs[end][1], r2.xs[end][2], r2.fs[end], r2.grad_norms[end])
plot_trajectory(rastrigin, r2, (-5.2,5.2), (-5.2,5.2), "dfp_results/rastrigin.png", title="Rastrigin")

r3 = dfp_optimize(schwefel, grad_schwefel, vec2(420.0,420.0), max_iter=500, eps=1e-8, α0=0.1, name="Schwefel")
@printf("Schwefel iter=%d converged=%s x=(%.8f %.8f) f=%.2e ||g||=%.2e\n", length(r3.xs)-1, r3.converged, r3.xs[end][1], r3.xs[end][2], r3.fs[end], r3.grad_norms[end])
plot_trajectory(schwefel, r3, (-500,500), (-500,500), "dfp_results/schwefel.png", title="Schwefel")

rq = dfp_quad(vec2(1.0,1.0))
@printf("Quadratic iter=%d x=(%.8f %.8f) f=%.2e\n", length(rq.xs)-1, rq.xs[end][1], rq.xs[end][2], rq.fs[end])
sumA = sum(rq.As)
sumB = sum(rq.Bs)
@printf("sumA-Hinv=%.2e sumB-I=%.2e\n", norm(sumA-Ginv), norm(sumB-Matrix{Float64}(I,2,2)))
plot_trajectory(quad_f, rq, (-0.5,1.5), (-0.5,1.5), "dfp_results/quadratic.png", title="Quadratic")

