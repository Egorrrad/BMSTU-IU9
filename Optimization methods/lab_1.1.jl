using Plots

# f(x) = x² - 10·cos(2πx) + 10 (вариант 32)
f(x) = x^2 - 10 * cos(2 * pi * x) + 10


# Метод Свенна для поиска интервала [a, b]
function svenn(f, x0, h)
    f0 = f(x0)
    if f(x0 + h) < f0
        x1, x2 = x0, x0 + h
    elseif f(x0 - h) < f0
        h = -h
        x1, x2 = x0, x0 + h
    else
        return x0 - h, x0 + h
    end
    while f(x2) < f(x1)
        h *= 2
        x1, x2 = x2, x2 + h
    end
    return min(x1, x2), max(x1, x2)
end

# Численная производная
num_der(f, x, d=1e-8) = (f(x + d) - f(x - d)) / (2 * d)

function log_unimodality(f, a, b)
    points = range(a, b, length=100)
    ders = [num_der(f, x) for x in points]
    sign_changes = sum(ders[i] * ders[i+1] < 0 for i in 1:length(ders)-1)
    println("--- ЛОГ УНИМОДАЛЬНОСТИ ---")
    println("Интервал [$a, $b]. Смен знака производной: $sign_changes")
    println("Функция унимодальна: $(sign_changes == 1)")
end

# Правило дождя
function log_rain_rule(f, x_min, eps)
    h = eps * 1.1
    f0 = f(x_min)
    left_ok = f(x_min - h) >= f0
    right_ok = f(x_min + h) >= f0
    println("--- ПРАВИЛО ДОЖДЯ ---")
    println("Слева выше: $left_ok, Справа выше: $right_ok")
    println("Минимум подтвержден: $(left_ok && right_ok)")
end

# 1. Перебор
function exhaustive_search(f, a, b, eps)
    n = Int(ceil((b - a) / eps))
    xs = range(a, b, length=n)
    fs = [f(x) for x in xs]
    idx = argmin(fs)
    return xs[idx], n, collect(xs), fs
end

# 2. Дихотомия
function dichotomy_search(f, a, b, eps)
    delta = eps / 3
    hx, hy, iters = Float64[], Float64[], 0
    while (b - a) > eps
        iters += 1
        x1, x2 = (a+b)/2 - delta, (a+b)/2 + delta
        f1, f2 = f(x1), f(x2)
        push!(hx, x1, x2); push!(hy, f1, f2)
        if f1 < f2 b = x2 else a = x1 end
    end
    return (a + b) / 2, iters, hx, hy
end

# 3. Золотое сечение
function golden_search(f, a, b, eps)
    phi = (sqrt(5) - 1) / 2
    x1, x2 = b - phi*(b-a), a + phi*(b-a)
    f1, f2 = f(x1), f(x2)
    hx, hy, iters = [x1, x2], [f1, f2], 1
    while (b - a) > eps
        iters += 1
        if f1 < f2
            b = x2; x2 = x1; f2 = f1
            x1 = b - phi*(b-a); f1 = f(x1)
        else
            a = x1; x1 = x2; f1 = f2
            x2 = a + phi*(b-a); f2 = f(x2)
        end
        push!(hx, x1); push!(hy, f1)
    end
    return (a + b) / 2, iters, hx, hy
end

# 4. Фибоначчи
fib(n) = n <= 1 ? n : fib(n-1) + fib(n-2)

function fibonacci_search(f, a, b, eps)
    n = 1
    while fib(n + 2) < (b - a) / eps n += 1 end
    x1 = a + (fib(n)/fib(n+2))*(b-a)
    x2 = a + (fib(n+1)/fib(n+2))*(b-a)
    f1, f2 = f(x1), f(x2)
    hx, hy = [x1, x2], [f1, f2]
    for k in 1:n-1
        if f1 > f2
            a = x1; x1 = x2; f1 = f2
            x2 = a + (fib(n-k+1)/fib(n-k+2))*(b-a); f2 = f(x2)
        else
            b = x2; x2 = x1; f2 = f1
            x1 = a + (fib(n-k)/fib(n-k+2))*(b-a); f1 = f(x1)
        end
        push!(hx, x1); push!(hy, f1)
    end
    return (a + b) / 2, n, hx, hy
end


eps = 0.005
a_s, b_s = svenn(f, 0.1, 0.05)

log_unimodality(f, a_s, b_s)

res_ex = exhaustive_search(f, a_s, b_s, eps)
res_di = dichotomy_search(f, a_s, b_s, eps)
res_gs = golden_search(f, a_s, b_s, eps)
res_fi = fibonacci_search(f, a_s, b_s, eps)

println("\n--- КОЛИЧЕСТВО ИТЕРАЦИЙ ---")
println("Перебор: $(res_ex[2])")
println("Дихотомия: $(res_di[2])")
println("Золотое сечение: $(res_gs[2])")
println("Фибоначчи: $(res_fi[2])")

log_rain_rule(f, res_gs[1], eps)

# --- 3. ГРАФИК ---
p = plot(f, a_s - 0.05, b_s + 0.05, lw=2, label="f(x)", title="Лабораторная №2")
colors = [:red, :blue, :green, :orange]
data = [res_ex, res_di, res_gs, res_fi]
names = ["Перебор", "Дихотомия", "Золотое сеч.", "Фибоначчи"]

for i in 1:4
    # Точки релаксационной последовательности (x, y)
    scatter!(data[i][3], data[i][4], color=colors[i], label=names[i], markersize=3)
    
    # Точки на оси X (проекция x)
    scatter!(data[i][3], fill(-1.0, length(data[i][3])), color=colors[i], markershape=:vline, label="", alpha=0.3)
    
    # Точки на оси Y (проекция f(x))
    scatter!(fill(a_s-0.04, length(data[i][4])), data[i][4], color=colors[i], markershape=:hline, label="", alpha=0.3)
end

# Интервал точности
vspan!([res_gs[1]-eps, res_gs[1]+eps], color=:yellow, alpha=0.2, label="Интервал точности")

display(p)
