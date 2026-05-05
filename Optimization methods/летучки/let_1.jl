
function svenn(f, x0, h)
    # Инициализация начальных точек
    x1 = x0
    x2 = x0 + h
    # Если значение в x2 больше, чем в x1, меняем направление шага
    if f(x2) > f(x1)
        h = -h
        x2 = x0 + h
    end
    # Пока значение в x2 меньше, чем в x1, продолжаем удваивать шаг и двигаться
    while f(x2) < f(x1)
        x1 = x2
        x2 = x1 + h
        h = 2 * h
    end
    # Возвращаем интервал [a, b] как минимум и максимум из x1 и x2
    a = min(x1, x2)
    b = max(x1, x2)
    return a, b
end

function fibonacci_search(f, a, b, eps)
    # Определяем количество итераций n на основе чисел Фибоначчи
    n = 1
    while fib(n + 2) < (b - a) / eps
        n += 1
    end
    k = 1
    # Вычисляем начальные точки x1 и x2 с использованием коэффициентов Фибоначчи
    x1 = a + (b - a) * fib(n) / fib(n + 2)
    x2 = a + (b - a) * fib(n + 1) / fib(n + 2)
    f1 = f(x1)
    f2 = f(x2)

    while k < n
        if f1 > f2
            # Если f1 > f2, сдвигаем a к x1 и пересчитываем x2
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) * fib(n - k + 1) / fib(n - k + 2)
            f2 = f(x2)
        else
            # Иначе сдвигаем b к x2 и пересчитываем x1
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (b - a) * fib(n - k) / fib(n - k + 2)
            f1 = f(x1)
        end
        k += 1
    end
    # выбираем лучший минимум
    candidates = [a, b, (a + b) / 2, x1, x2]
    x_min = candidates[1]
    for c in candidates
        if f(c) < f(x_min)
            x_min = c
        end
    end
    
    println("Точка минимума: ", x_min)
    println("Заданная точность: ", eps)
    return x_min, eps
end

# Функция для вычисления n-го числа Фибоначчи
function fib(n)
    if n <= 1
        return n
    end
    # Итеративный расчет
    a, b = 0, 1
    for _ in 2:n
        a, b = b, a + b
    end
    return b
end

# Числовая первая производная функции f в точке x
function num_derivative(f, x, delta=1e-7)
    return (f(x + delta) - f(x - delta)) / (2 * delta)
end

# Числовая вторая производная функции f в точке x
function num_second_derivative(f, x, delta=1e-6)
    return (f(x + delta) - 2 * f(x) + f(x - delta)) / (delta^2)
end

# Проверка унимодальности функции на интервале [a, b] численно
function prove_unimodality(f, a, b, n_points=100)
    # Вычисляем шаг и инициализируем счетчик смен знака
    step = (b - a) / (n_points - 1)
    sign_changes = 0
    prev_der = num_derivative(f, a)
    # Проверяем производные в точках интервала
    for i in 2:n_points
        x = a + (i - 1) * step
        der = num_derivative(f, x)
        # Считаем смену знака производной
        if (prev_der < 0 && der >= 0) || (prev_der > 0 && der <= 0)
            sign_changes += 1
        end
        prev_der = der
    end
    # Унимодальность, если смена знака ровно один раз
    unimodal = (sign_changes == 1)
    println("Численная проверка унимодальности на [", a, ", ", b, "]: смена знака первой производной ", sign_changes, " раз. Унимодальность: ", unimodal)
    return unimodal
end

function prove_minimum_rule(f, x_min, precision=1e-5)
    # Определяем delta для проверки
    delta = max(1e-10, precision * 0.5)
    # Вычисляем вторую производную
    fpp = num_second_derivative(f, x_min)
    f0 = f(x_min)
    # Проверяем правило дождя: значения слева и справа >= f(x_min)
    rain_left = f(x_min - delta) >= f0 - 1e-12
    rain_right = f(x_min + delta) >= f0 - 1e-12
    is_min_rain = rain_left && rain_right
    is_min = is_min_rain
    
    println("Вторая производная в точке (численно): ", fpp)
    println("Правило дождя: f(x*-h) >= f(x*), f(x*+h) >= f(x*) при h=", delta, " : ", rain_left, ", ", rain_right)
    println("Точка является минимумом (не максимум и не перегиб): ", is_min)
    return is_min
end

f(x) = x^3 - 3 * x

a, b = svenn(f, 0.0, 0.1)
println("Метод Свенна: интервал [", a, ", ", b, "]")
x_min, precision = fibonacci_search(f, a, b, 1e-10)
prove_unimodality(f, a, b)
prove_minimum_rule(f, x_min, precision)

