import Pkg
Pkg.add(["Plots", "PlotlyJS", "FileIO", "ImageIO", "ImageTransformations", "Colors", "ImageMagick"])

const LAB19_ROOT = pwd()
using Random, Statistics, LinearAlgebra
using FileIO, ImageIO, ImageTransformations, Colors, Plots
plotlyjs()

default(; size = (700, 400), show = false)

const SAVE_DIR = mkpath(joinpath(LAB19_ROOT, "lab19"))
plot_cache_dir() = mkpath(joinpath(LAB19_ROOT, ".plot_display_cache"))
plot_counter = Ref(0)

function showplot(p = Plots.current(); filename::String = "")
    d = plot_cache_dir()
    path = joinpath(d, string(time_ns(), ".png"))
    try
        savefig(p, path)
        display(MIME("image/png"), read(path))
        if isempty(filename)
            plot_counter[] += 1
            filename = string("plot_", lpad(plot_counter[], 3, "0"), ".png")
        end
        savefig(p, joinpath(SAVE_DIR, filename))
    finally
        rm(path; force = true)
    end
    return nothing
end

const IMG_SIZE = 150
const CANVAS_REF = 120.0

mod8(dir::Int) = mod(dir, 8)

function calculate_stems(g::AbstractVector{Int})
    g1,g2,g3,g4,g5,g6,g7 = g[1],g[2],g[3],g[4],g[5],g[6],g[7]
    [(0,g1),(g2,g3),(g4,0),(g5,-g6),(0,-g7),(-g5,-g6),(-g4,0),(-g2,g3)]
end

function render_creature!(segments, L::Int, stems, dir::Int, ox::Float64, oy::Float64)
    L < 1 && return segments
    d = mod8(dir); sx, sy = stems[d+1]
    nx = ox + L*sx; ny = oy + L*sy
    push!(segments, (ox, oy, nx, ny))
    if L > 1
        render_creature!(segments, L-1, stems, dir+1, nx, ny)
        render_creature!(segments, L-1, stems, dir-1, nx, ny)
    end
    return segments
end

function segment_bounds(segs)
    minx = miny = Inf; maxx = maxy = -Inf
    for (x0,y0,x1,y1) in segs, (x,y) in ((x0,y0),(x1,y1))
        minx=min(minx,x); maxx=max(maxx,x); miny=min(miny,y); maxy=max(maxy,y)
    end
    isempty(segs) ? (0.0,0.0,0.0,0.0) : (minx,maxx,miny,maxy)
end

function draw_line!(img::AbstractMatrix{<:Real}, x0::Int, y0::Int, x1::Int, y1::Int, val)
    dx=abs(x1-x0); dy=-abs(y1-y0); sx=x0<x1 ? 1 : -1; sy=y0<y1 ? 1 : -1
    err=dx+dy; x,y=x0,y0; h,w=size(img)
    while true
        1<=x<=w && 1<=y<=h && (img[y,x]=val)
        (x==x1 && y==y1) && break
        e2=2err; e2>=dy && (err+=dy; x+=sx); e2<=dx && (err+=dx; y+=sy)
    end
    return img
end

function dilate_binary_max!(img::Matrix{Float64}, passes::Int)
    passes <= 0 && return img
    h,w=size(img)
    for _ in 1:passes
        snap=copy(img); fill!(img,0.0)
        for j in 1:h, i in 1:w
            snap[j,i]<0.5 && continue
            for dj in -1:1, di in -1:1
                jj,ii=j+dj,i+di
                1<=jj<=h && 1<=ii<=w && (img[jj,ii]=1.0)
            end
        end
    end
    return img
end

function biomorph_raster(genes::Vector{Int}; size::Int=IMG_SIZE, canvas::Float64=CANVAS_REF, line_half_width::Int=0)
    @assert length(genes)==16
    Lrec=genes[16]
    stems=Tuple{Int,Int}[(Int(sx),Int(sy)) for (sx,sy) in calculate_stems(genes)]
    segs=Tuple{Float64,Float64,Float64,Float64}[]
    render_creature!(segs, Lrec, stems, 0, 0.0, 0.0)
    minx,maxx,miny,maxy=segment_bounds(segs)
    halfw=max(abs(minx),abs(maxx),1e-6); halfh=max(abs(miny),abs(maxy),1e-6)
    factor=canvas/(2*max(halfw,halfh))
    img=zeros(Float64,size,size); cx=(size+1)/2; cy=(size+1)/2
    for (x0,y0,x1,y1) in segs
        draw_line!(img,round(Int,cx+factor*x0),round(Int,cy-factor*y0),
                       round(Int,cx+factor*x1),round(Int,cy-factor*y1),1.0)
    end
    dilate_binary_max!(img, line_half_width)
    return img
end

function random_genotype(rng::AbstractRNG=Random.default_rng())
    g=Vector{Int}(undef,16)
    for i in 1:15; g[i]=rand(rng,-9:9); end
    g[16]=rand(rng,2:12); return g
end

function clamp_gene!(g::Vector{Int}, i::Int)
    g[i] = i<=15 ? clamp(g[i],-9,9) : clamp(g[16],2,12); return g
end

function mutate_one(g::Vector{Int}, rng::AbstractRNG=Random.default_rng())
    child=copy(g); i=rand(rng,1:16)
    child[i]+=rand(rng,(-1,1)); clamp_gene!(child,i); return child
end

function load_target_gray(path::AbstractString; target_size::Int=IMG_SIZE)
    isfile(path) || error("Файл не найден: $path")
    img=load(path); g=Float64.(Gray.(img))
    (size(g,1)!=target_size || size(g,2)!=target_size) &&
        (g=Float64.(imresize(Gray.(img),(target_size,target_size))))
    return g
end

fitness_euclidean(a,b) = -sqrt(sum(abs2, a.-b))
fitness_manhattan(a,b) = -sum(abs.(a.-b))

function fitness_ncc(a::AbstractMatrix{Float64}, b::AbstractMatrix{Float64})
    aa=a.-mean(a); bb=b.-mean(b)
    da=sqrt(sum(abs2,aa)); db=sqrt(sum(abs2,bb))
    (da<1e-12 || db<1e-12) && return 0.0
    dot(vec(aa),vec(bb))/(da*db)
end

@enum FitnessMode EuclideanDist ManhattanDist NCC

function fitness(mode::FitnessMode, pheno, target)
    mode==EuclideanDist && return fitness_euclidean(pheno,target)
    mode==ManhattanDist && return fitness_manhattan(pheno,target)
    return fitness_ncc(pheno,target)
end

is_better(::FitnessMode, newf::Float64, oldf::Float64) = newf > oldf

function evolve_biomorph(
    target::Matrix{Float64};
    N::Int=16, max_stagnation::Int=10, mode::FitnessMode=EuclideanDist,
    rng::AbstractRNG=Random.default_rng(), init=nothing, line_half_width::Int=0,
    sa_temperature::Float64=0.0, sa_decay::Float64=0.995, sa_min::Float64=1e-4,
    max_generations::Int=typemax(Int), fitness_goal=nothing, early_stop_offspring::Bool=false,
)
    parent = init===nothing ? random_genotype(rng) : copy(init)
    pheno_parent = biomorph_raster(parent; line_half_width)
    f_parent = fitness(mode, pheno_parent, target)
    history_f = Float64[f_parent]; history_genes = Vector{Int}[copy(parent)]
    stagnation=0; generation=0; T_sa=sa_temperature
    goal_reached() = fitness_goal===nothing ? false : !is_better(mode,fitness_goal,f_parent)
    while stagnation < max_stagnation
        goal_reached() && break
        generation+=1; best_g=parent; best_f=nothing; best_p=pheno_parent
        for _ in 1:N
            child=mutate_one(parent,rng); p=biomorph_raster(child;line_half_width)
            fc=fitness(mode,p,target)
            if best_f===nothing || is_better(mode,fc,best_f)
                best_f=fc; best_g=child; best_p=p
            end
            early_stop_offspring && is_better(mode,best_f,f_parent) && break
        end
        improved = is_better(mode,best_f,f_parent)
        accept_worse = !improved && T_sa>0.0 && rand(rng)<exp((best_f-f_parent)/T_sa)
        if improved || accept_worse
            parent=best_g; f_parent=best_f; pheno_parent=best_p; stagnation=0
        else
            stagnation+=1
        end
        T_sa>0.0 && (T_sa=max(sa_min,T_sa*sa_decay))
        push!(history_f,f_parent); push!(history_genes,copy(parent))
        goal_reached() && break
        generation>=max_generations && break
    end
    return (;parent,f_parent,pheno_parent,history_f,history_genes,generation,stagnation,line_half_width)
end

function gif_biomorph_frames(plot_i::Function, out_path::AbstractString, nframes::Int; fps::Real=2)
    d = mktempdir(plot_cache_dir(); prefix="lab19_gif_")
    anim = Animation(d, String[])
    prev_show = Plots.default(:show); default(show=false)
    try
        for i in 1:nframes; plot_i(i); frame(anim); end
        gif(anim, String(out_path); fps=fps)
    finally
        default(show=prev_show); isdir(d) && rm(d; recursive=true)
    end
    return out_path
end

function show_gif_in_notebook(path::AbstractString; title::AbstractString="")
    isempty(title) || println(title)
    display(MIME("image/gif"), read(path))
    return nothing
end

TARGET_PATH = joinpath(LAB19_ROOT, "tree.jpg")
target = load_target_gray(TARGET_PATH)
@info "Эталон загружен" TARGET_PATH
showplot(heatmap(Gray.(target); aspect_ratio=1, title="Эталон tree", size=(400,400)); filename="target_tree.png")

Random.seed!(42)
N_POP=24; FITNESS=EuclideanDist; LINE_HALF_WIDTH=0
result = evolve_biomorph(target; N=N_POP, max_stagnation=10, mode=FITNESS, line_half_width=LINE_HALF_WIDTH)
@info "Поколений" result.generation; @info "Fitness" result.f_parent

p1 = heatmap(Gray.(result.pheno_parent); aspect_ratio=1, title="Лучший биоморф")
p2 = plot(result.history_f; legend=false, xlabel="поколение", ylabel="fitness", title="Динамика")
showplot(plot(p1, p2; layout=(1,2), size=(900,400)); filename="biomorph_run_tree.png")
println("Генотип: ", result.parent)

evol_gif_path = joinpath(LAB19_ROOT, "biomorph_evolution_tree.gif")
histg=result.history_genes; Ne=length(histg)
gif_biomorph_frames(evol_gif_path, Ne; fps=2) do i
    heatmap(Gray.(biomorph_raster(histg[i]; line_half_width=result.line_half_width));
            aspect_ratio=1, title="поколение $i/$Ne", colorbar=false)
end
show_gif_in_notebook(evol_gif_path; title="Эволюция tree")

g0 = random_genotype(Random.Xoshiro(1)); figs=[]
for mode in (EuclideanDist, ManhattanDist, NCC)
    Random.seed!(1)
    r = evolve_biomorph(target; N=16, max_stagnation=10, mode, init=g0, line_half_width=LINE_HALF_WIDTH)
    push!(figs, heatmap(Gray.(r.pheno_parent); title=string(mode), aspect_ratio=1))
end
showplot(plot(figs...; layout=(1,3), size=(900,320)); filename="metrics_tree.png")

TARGET_PATH = joinpath(LAB19_ROOT, "bat.jpg")
target = load_target_gray(TARGET_PATH)
@info "Эталон загружен" TARGET_PATH
showplot(heatmap(Gray.(target); aspect_ratio=1, title="Эталон bat", size=(400,400)); filename="target_bat.png")

Random.seed!(42)
result = evolve_biomorph(target; N=N_POP, max_stagnation=10, mode=FITNESS, line_half_width=LINE_HALF_WIDTH)
@info "Поколений (bat)" result.generation; @info "Fitness (bat)" result.f_parent

p1 = heatmap(Gray.(result.pheno_parent); aspect_ratio=1, title="Лучший биоморф (bat)")
p2 = plot(result.history_f; legend=false, xlabel="поколение", ylabel="fitness", title="Динамика")
showplot(plot(p1, p2; layout=(1,2), size=(900,400)); filename="biomorph_run_bat.png")

evol_gif_path = joinpath(LAB19_ROOT, "biomorph_evolution_bat.gif")
histg=result.history_genes; Ne=length(histg)
gif_biomorph_frames(evol_gif_path, Ne; fps=2) do i
    heatmap(Gray.(biomorph_raster(histg[i]; line_half_width=result.line_half_width));
            aspect_ratio=1, title="поколение $i/$Ne", colorbar=false)
end
show_gif_in_notebook(evol_gif_path; title="Эволюция bat")

g0=random_genotype(Random.Xoshiro(1)); figs=[]
for mode in (EuclideanDist, ManhattanDist, NCC)
    Random.seed!(1)
    r=evolve_biomorph(target; N=16, max_stagnation=10, mode, init=g0, line_half_width=LINE_HALF_WIDTH)
    push!(figs, heatmap(Gray.(r.pheno_parent); title=string(mode), aspect_ratio=1))
end
showplot(plot(figs...; layout=(1,3), size=(900,320)); filename="metrics_bat.png")

synth_ref_genes = Int[2,1,-3,4,2,-5,0,0,0,0,0,0,0,0,0,9]
lw = LINE_HALF_WIDTH
target = biomorph_raster(synth_ref_genes; line_half_width=lw)
showplot(heatmap(Gray.(target); aspect_ratio=1, title="Синтетический эталон", size=(400,400)); filename="target_synth.png")

Random.seed!(42)
Ft=NCC; synth_fitness_goal=fitness(Ft,target,target)
synth_start=Int[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9]
SYNTH_GIF_FRAMES=24

result = evolve_biomorph(target; N=N_POP, max_stagnation=12, mode=Ft, line_half_width=lw,
    init=synth_start, sa_temperature=5.0, sa_decay=0.997,
    max_generations=52, fitness_goal=synth_fitness_goal, early_stop_offspring=true)
@info "Синтетика: поколений" result.generation; @info "Синтетика: fitness" result.f_parent

p1=heatmap(Gray.(result.pheno_parent); aspect_ratio=1, title="Лучший биоморф (синтетика)")
p2=plot(result.history_f; legend=false, xlabel="поколение", ylabel="fitness", title="Динамика")
showplot(plot(p1,p2; layout=(1,2), size=(900,400)); filename="biomorph_run_synth.png")

evol_gif_path=joinpath(LAB19_ROOT,"biomorph_evolution_synth.gif")
histg=result.history_genes; Ne=length(histg)
gif_idx = Ne<=SYNTH_GIF_FRAMES ? collect(1:Ne) : unique(round.(Int, range(1,Ne;length=SYNTH_GIF_FRAMES)))
gif_biomorph_frames(evol_gif_path, length(gif_idx); fps=2) do i
    j=gif_idx[i]
    heatmap(Gray.(biomorph_raster(histg[j]; line_half_width=result.line_half_width));
            aspect_ratio=1, size=(360,360), title="поколение $j/$Ne", colorbar=false)
end
show_gif_in_notebook(evol_gif_path; title="Эволюция к синтетическому эталону")

N_LINEAGES=5; WARMUP_MUTATIONS=5
lw_r=LINE_HALF_WIDTH
ref_genes=Int[2,1,-3,4,2,-5,0,0,0,0,0,0,0,0,0,9]
target_r=biomorph_raster(ref_genes; line_half_width=lw_r)
line0=Int[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9]

inits_r = map(1:N_LINEAGES) do c
    g=copy(line0); rng_w=Random.Xoshiro(10_000+c)
    for _ in 1:WARMUP_MUTATIONS; g=mutate_one(g,rng_w); end; g
end

runs_r = map(1:N_LINEAGES) do c
    evolve_biomorph(target_r; N=N_POP, max_stagnation=18, max_generations=45,
        mode=NCC, init=inits_r[c], rng=Random.Xoshiro(20_000+c), line_half_width=lw_r)
end
for c in 1:N_LINEAGES; @info "цепочка $c" generations=runs_r[c].generation fitness=runs_r[c].f_parent; end

function _biomorph_hm(img; px=120)
    heatmap(Gray.(img); aspect_ratio=1, colorbar=false, ticks=false,
            showaxis=false, grid=false, legend=false, framestyle=:none, size=(px,px))
end

STRIP_CHAIN=1; hist_strip=runs_r[STRIP_CHAIN].history_genes; Ls=length(hist_strip); HM=180
strip_cells=[_biomorph_hm(target_r; px=HM) for _ in 1:Ls]
append!(strip_cells,[_biomorph_hm(biomorph_raster(hist_strip[i];line_half_width=lw_r);px=HM) for i in 1:Ls])
showplot(plot(strip_cells...; layout=(2,Ls), size=(Ls*HM,2*HM)); filename="five_lineages_strip.png")

hf0=runs_r[1].history_f
p_fit=plot(0:length(hf0)-1, hf0; legend=true, label="1", xlabel="поколение", ylabel="fitness", title="5 цепочек")
for c in 2:N_LINEAGES
    hf=runs_r[c].history_f; plot!(p_fit, 0:length(hf)-1, hf; label=string(c))
end
showplot(p_fit; filename="five_lineages_fitness.png")

best_c=argmax([r.f_parent for r in runs_r])
@info "Лучшая цепочка" best_c fitness=runs_r[best_c].f_parent generations=runs_r[best_c].generation
