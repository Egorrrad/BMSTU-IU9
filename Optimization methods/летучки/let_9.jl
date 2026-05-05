import Pkg
Pkg.add(["Plots", "Optim"])

using Plots, Optim
gr(dpi=150, size=(700,450))
mkpath("let9")

a, b = 0.0, 1.0
xs = range(a, b, length=600)

f1(x) = 5 - 24*x + 17*x^2 - (11/3)*x^3 + (1/4)*x^4
f2(x) = x^2 - 10*cos(2π*x) + 10

f1v = f1.(xs); f2v = f2.(xs)
f1min, f1max = minimum(f1v), maximum(f1v)
f2min, f2max = minimum(f2v), maximum(f2v)

f1n(x) = (f1(x) - f1min) / (f1max - f1min)
f2n(x) = (f2(x) - f2min) / (f2max - f2min)

p1 = plot(collect(xs), f1v, label="f₁(x)=x²", xlabel="x", ylabel="f₁",
    title="f₁(x) → min", lw=2.5, color=:steelblue,
    fillrange=0, fillalpha=0.08, fillcolor=:steelblue)
scatter!([0.0],[0.0], label="min f₁", color=:red, ms=9)
savefig(p1, "let9/graph1_f1.png"); display(p1)

p2 = plot(collect(xs), f2v, label="f₂(x)=(1-x)²", xlabel="x", ylabel="f₂",
    title="f₂(x) → min", lw=2.5, color=:darkorange,
    fillrange=0, fillalpha=0.08, fillcolor=:darkorange)
scatter!([1.0],[0.0], label="min f₂", color=:red, ms=9)
savefig(p2, "let9/graph2_f2.png"); display(p2)

p3 = plot(f1v, f2v, label="Фронт Парето", xlabel="f₁", ylabel="f₂",
    title="Фронт Парето", lw=3, color=:purple)
scatter!([f1min],[f2min], label="Идеальная точка", color=:gold, ms=12, markershape=:star5)
scatter!([f1max],[f2max], label="Надир", color=:darkred, ms=9, markershape=:dtriangle)
for xi in [0.25, 0.5, 0.75]
    scatter!([f1(xi)],[f2(xi)], label="x=$(xi)", ms=7, color=:steelblue)
end
savefig(p3, "let9/graph3_pareto.png"); display(p3)

F_w(x, w1, w2) = w1 * f1n(x)^2 + w2 * f2n(x)^2

configs = [
    (0.5, 0.5, :steelblue,  "w=(0.5,0.5)"),
    (0.8, 0.2, :red,        "w=(0.8,0.2)"),
    (0.2, 0.8, :darkorange, "w=(0.2,0.8)"),
    (1.0, 0.0, :darkgreen,  "w=(1.0,0.0)"),
]

p4 = plot(f1n.(xs), f2n.(xs), label="Фронт Парето (норм.)",
    xlabel="f̃₁", ylabel="f̃₂", title="Метод идеальной точки",
    lw=2.5, color=:purple, linestyle=:dash,
    xlims=(-0.05,1.15), ylims=(-0.05,1.15))
scatter!([0.0],[0.0], label="X*=(0,0)", ms=14, color=:gold, markershape=:star5)

for (w1, w2, clr, lbl) in configs
    res = optimize(x -> F_w(x, w1, w2), a, b, Brent())
    xo = Optim.minimizer(res)
    plot!([f1n(xo),0.0],[f2n(xo),0.0], label=false, color=clr, lw=1.2, ls=:dot, alpha=0.7)
    scatter!([f1n(xo)],[f2n(xo)], label=lbl*" x*=$(round(xo,digits=3))", ms=9, color=clr)
end
savefig(p4, "let9/graph4_ideal_point.png"); display(p4)
println(sum(first.(configs)))

