
using LinearAlgebra, Printf, Plots, Plots.PlotMeasures
mkpath("let3_output")

quad(x,α,β) = α*x[1]^2 + β*x[2]^2
grad_quad(x,α,β) = [2α*x[1], 2β*x[2]]

rosen(x) = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2
grad_rosen(x) = [-2*(1-x[1])-400*x[1]*(x[2]-x[1]^2), 200*(x[2]-x[1]^2)]

rastr(x) = 20 + sum(xi^2-10*cos(2π*xi) for xi in x)
grad_rastr(x) = [2*xi+20π*sin(2π*xi) for xi in x]

schwef(x) = 418.9829*2 - sum(xi*sin(sqrt(abs(xi))) for xi in x)
grad_schwef(x) = begin
    g = similar(x, Float64)
    for i in eachindex(x)
        xi=x[i]; sq=sqrt(abs(xi)+1e-12)
        g[i] = -sin(sq) - xi*cos(sq)/(2sq+1e-12)*sign(xi)
    end; g
end

function golden(g; b=5.0, tol=1e-9, maxit=500)
    φ = (√5-1)/2; a0,b0 = 0.0,b
    x1=b0-φ*(b0-a0); x2=a0+φ*(b0-a0); f1,f2=g(x1),g(x2)
    for _ in 1:maxit
        abs(b0-a0)<tol && break
        if f1<f2; b0,x2,f2=x2,x1,f1; x1=b0-φ*(b0-a0); f1=g(x1)
        else; a0,x1,f1=x1,x2,f2; x2=a0+φ*(b0-a0); f2=g(x2); end
    end; (a0+b0)/2
end

function gradient_descent(f,∇f,x0; α=0.01, tol=1e-6, maxit=50_000)
    x=float.(x0); tr=[copy(x)]; fv=[f(x)]
    for _ in 1:maxit
        g=∇f(x); norm(g)≤tol && break
        αk=α; xn=x-αk*g
        while f(xn)≥f(x) && αk>1e-15; αk/=2; xn=x-αk*g; end
        x=xn; push!(tr,copy(x)); push!(fv,f(x))
    end; x,fv,tr
end

function steepest_descent(f,∇f,x0; tol=1e-6, maxit=50_000, gs_b=5.0)
    x=float.(x0); tr=[copy(x)]; fv=[f(x)]
    for _ in 1:maxit
        g=∇f(x); norm(g)≤tol && break
        y=-g; αk=golden(a->f(x+a*y); b=gs_b)
        x=x+αk*y; push!(tr,copy(x)); push!(fv,f(x))
    end; x,fv,tr
end

function conjugate_gradients(f,∇f,x0; tol=1e-6, maxit=50_000, N=50, gs_b=5.0)
    x=float.(x0); tr=[copy(x)]; fv=[f(x)]
    g_prev=zeros(length(x0)); p=zeros(length(x0))
    for k in 1:maxit
        g=∇f(x); norm(g)≤tol && break
        if k==1 || mod(k,N)==0; p=copy(g)
        else; β=dot(g,g)/(dot(g_prev,g_prev)+1e-30); p=g+β*p; end
        αk=golden(a->f(x-a*p); b=gs_b)
        g_prev=copy(g); x=x-αk*p; push!(tr,copy(x)); push!(fv,f(x))
    end; x,fv,tr
end

const COLORS = [:red, :dodgerblue, :limegreen]
const NAMES = ["Градиентный спуск","Наискорейший спуск","Сопряжённые градиенты"]

function add_traj!(plt, tr, f, col; is3d=false, lw=1.5, ms=3)
    tx=[p[1] for p in tr]; ty=[p[2] for p in tr]
    if is3d
        tz=[f(p) for p in tr]
        plot!(plt, tx, ty, tz; color=col, lw=lw, label="")
        length(tx)>2 && scatter!(plt, tx[2:end-1], ty[2:end-1], tz[2:end-1];
            color=col, ms=ms, markerstrokewidth=0, label="")
        scatter!(plt, [tx[end]], [ty[end]], [tz[end]]; color=col, ms=ms+3,
            markershape=:hexagon, markerstrokecolor=:black, markerstrokewidth=0.8, label="")
    else
        plot!(plt, tx, ty; color=col, lw=lw, label="")
        length(tx)>2 && scatter!(plt, tx[2:end-1], ty[2:end-1];
            color=col, ms=ms, markerstrokewidth=0, label="")
        scatter!(plt, [tx[end]], [ty[end]]; color=col, ms=ms+3,
            markershape=:hexagon, markerstrokecolor=:black, markerstrokewidth=0.8, label="")
    end
end

function add_start!(plt, x0, f; is3d=false)
    if is3d
        scatter!(plt,[x0[1]],[x0[2]],[f(x0)]; color=:white, ms=6, markershape=:circle,
            markerstrokecolor=:black, markerstrokewidth=1.8, label="Старт")
    else
        scatter!(plt,[x0[1]],[x0[2]]; color=:white, ms=6, markershape=:circle,
            markerstrokecolor=:black, markerstrokewidth=1.8, label="Старт")
    end
end

function make_grid(f, xl, yl; n=300)
    xs=range(xl[1],xl[2],n); ys=range(yl[1],yl[2],n)
    Z=[f([xi,yi]) for yi in ys, xi in xs]
    xs,ys,Z
end

function print_table(title, rows)
    println("\n","═"^66,"\n $title\n","═"^66)
    @printf(" %-18s %-26s %12s %8s\n","Функция","Метод","f*","Итерации")
    println(" ","─"^62)
    prev=""
    for (fn,mn,fs,ni) in rows
        tag = fn!=prev ? fn : ""
        @printf(" %-18s %-26s %12.4e %8d\n", tag, mn, fs, ni)
        fn!=prev && prev!="" && println(" ","─"^62)
        prev=fn
    end
    println(" ","─"^62)
end


# КВАДРАТИЧНЫЕ — x0=(100,100), β ∈ {1,10,20}

x0_quad = [100.0,100.0]
betas = [1,10,20]
lstyles = [:solid,:dash,:dot]
quad_rows=[]; quad_trs=[]
for (vi,beta) in enumerate(betas)
    fi = x->quad(x,1,beta); gfi = x->grad_quad(x,1,beta)
    xg,fg,trg = gradient_descent(fi,gfi,x0_quad; α=0.02, tol=1e-8, maxit=50_000)
    xs_,fs,trs = steepest_descent(fi,gfi,x0_quad; tol=1e-8, maxit=50_000, gs_b=300.0)
    xc,fc,trc = conjugate_gradients(fi,gfi,x0_quad; tol=1e-8, maxit=50_000, gs_b=300.0)
    for (tr,fv,col,mn) in [(trg,fg,COLORS[1],NAMES[1]),(trs,fs,COLORS[2],NAMES[2]),(trc,fc,COLORS[3],NAMES[3])]
        push!(quad_trs,(tr,fv,col,lstyles[vi],fi)); push!(quad_rows,("α=1,β=$beta",mn,last(fv),length(fv)-1))
    end
end
fi_ref=x->quad(x,1,20); xl_q=(-105.,105.); yl_q=(-105.,105.)
xs_q,ys_q,Z_q = make_grid(fi_ref,xl_q,yl_q)
p3q = surface(xs_q,ys_q,Z_q; color=:viridis, alpha=0.6, colorbar=false,
    xlabel="x₁",ylabel="x₂",zlabel="f",title="3D",camera=(28,225))
p2q = contourf(xs_q,ys_q,Z_q; levels=50,color=:viridis,colorbar=false,
    xlabel="x₁",ylabel="x₂",title="2D — траектории",
    xlims=xl_q,ylims=yl_q)
for (mn,col) in zip(NAMES,COLORS)
    plot!(p2q,[],[]; color=col,lw=2,label=mn)
end
for (vi,beta) in enumerate(betas)
    plot!(p2q,[],[]; color=:gray50, linestyle=lstyles[vi], lw=1.3, label="β=$beta")
end
for (tr,fv,col,ls,fi) in quad_trs
    add_traj!(p3q,tr,fi,col; is3d=true, lw=1.1, ms=2)
    add_traj!(p2q,tr,fi,col; is3d=false, lw=1.1, ms=2)
end
add_start!(p3q,x0_quad,fi_ref; is3d=true)
add_start!(p2q,x0_quad,fi_ref)
plot!(p2q,legend=:topleft,legendfontsize=6,foreground_color_legend=:gray30)
fig_q=plot(p3q,p2q; layout=(1,2),size=(1400,600),
    plot_title="f(x)=αx₁²+βx₂², α=1, β∈{1,10,20}, x₀=(100,100)",
    plot_titlefontsize=11, margin=8mm)
savefig(fig_q,"let3_output/quad_all_variants.png"); display(fig_q)
print_table("КВАДРАТИЧНЫЕ f=x₁²+βx₂², x₀=(100,100)", quad_rows)


# НЕЛИНЕЙНЫЕ 
cases = [
    # Розенброк — классическая точка
    (rosen, grad_rosen, (-2.5,2.5), (-1.0,4.5), "Розенброк", "rosenbrock", 0.001, 1e-6, 30_000, true, 5, 1.0, [-1.2, 1.0]),
    # Растригин — другая точка
    (rastr, grad_rastr, (-5.12,5.12), (-5.12,5.12), "Растригин", "rastrigin", 0.015, 1e-5, 10_000, true, 50, 0.4, [4.0, 2.0]),
    # Швеффель — далеко от минимума
    (schwef, grad_schwef, (-500.,500.), (-500.,500.), "Швеффель", "schwefel", 2.0, 1e-4, 10_000, true, 50, 50.0, [-200.0, -400.0]),
]

nl_rows=[]
for (fi,gfi,xl,yl,name,fname,ag,tol,maxit,logz,N_r,gs_b,x0) in cases
    xg,fg,trg = gradient_descent(fi,gfi,x0; α=ag, tol=tol, maxit=maxit)
    xs_,fs,trs = steepest_descent(fi,gfi,x0; tol=tol, maxit=maxit, gs_b=gs_b)
    xc,fc,trc = conjugate_gradients(fi,gfi,x0; tol=tol, maxit=maxit, N=N_r, gs_b=gs_b)
    
    for (tr,fv,mn) in [(trg,fg,NAMES[1]),(trs,fs,NAMES[2]),(trc,fc,NAMES[3])]
        push!(nl_rows,(name,mn,last(fv),length(fv)-1))
    end
    
    xs_g,ys_g,Z = make_grid(fi,xl,yl)
    Zp = logz ? log1p.(max.(Z.-minimum(Z).+1e-12,0.0)) : Z
    p3 = surface(xs_g,ys_g,Zp; color=:viridis,alpha=0.55,colorbar=false,
        xlabel="x₁",ylabel="x₂",zlabel="f",title="3D",camera=(28,225))
    p2 = contourf(xs_g,ys_g,Zp; levels=60,color=:viridis,colorbar=false,
        xlabel="x₁",ylabel="x₂",title="2D — траектории",xlims=xl,ylims=yl)
    contour!(p2,xs_g,ys_g,Zp; levels=30,linecolor=:white,linewidth=0.3,colorbar=false)
    for (col,mn) in zip(COLORS,NAMES); plot!(p2,[],[]; color=col,lw=2,label=mn); end
    for (tr,fv,col) in [(trg,fg,COLORS[1]),(trs,fs,COLORS[2]),(trc,fc,COLORS[3])]
        add_traj!(p3,tr,fi,col; is3d=true, lw=1.8, ms=4)
        add_traj!(p2,tr,fi,col; is3d=false, lw=1.8, ms=4)
    end
    add_start!(p3,x0,fi; is3d=true); add_start!(p2,x0,fi)
    plot!(p2,legend=:topright,legendfontsize=7,foreground_color_legend=:gray30)
    fig_n=plot(p3,p2; layout=(1,2),size=(1400,600),
        plot_title="$name — 3 метода, x₀=$x0",
        plot_titlefontsize=11, margin=8mm)
    savefig(fig_n,"let3_output/$fname.png"); display(fig_n)
end

print_table("НЕЛИНЕЙНЫЕ (разные стартовые точки)", nl_rows)
println("\n✓ Все файлы в let3_output/")
println("   • Розенброк  → x₀ = [-1.2, 1.0]")
println("   • Растригин → x₀ = [4.0, 2.0]")
println("   • Швеффель  → x₀ = [-200.0, -400.0]")

