using LinearAlgebra

# Griewankova funkce
function f_griewank(x)
    x = x isa Number ? [x] : vec(x)  # Podpora skalárních i vektorových vstupů
    n = length(x)
    return 1 + sum(x .^ 2) / 4000 - prod(cos.(x ./ sqrt.(1:n)))
end


# Alternativní verze derivace Griewankovy funkce
function g_griewank(x)
    if x isa Number
        return x / 2000  # Pokud je skalár, vrátíme skalár
    else
        x = vec(x)  # Jinak zajistíme, že je to vektor
    end
    
    n = length(x)
    term1 = x / 2000
    if n == 0
        return term1
    end

    cos_term = cos.(x ./ sqrt.(1:n))
    sin_term = sin.(x ./ sqrt.(1:n)) ./ sqrt.(1:n)
    
    term2 = -prod(cos_term) .* sin_term
    
    return term1 + term2

end


# Projekce do přípustné množiny
function P(x, x_min, x_max)
    return clamp.(x, x_min, x_max)
end

# Gradientní metoda s projekcí
function optim(f, g, P, x; α=0.01, max_iter=10000)
    for _ in 1:max_iter
        y = x - α * g(x)
        x = P(y)
    end
    return x
end

# Funkce hledající všechna lokální minima
function generate_solutions(f, g, P, x_min, x_max, xs_init; atol=1e-5, α=0.01, max_iter=10000)
    solutions = []
    for i in 1:size(xs_init, 2)
        x_star = optim(f, g, x -> P(x, x_min, x_max), xs_init[:, i]; α=α, max_iter=max_iter)
        if !any(s -> norm(s - x_star) < atol, solutions)
            push!(solutions, x_star)
        end
    end
    return hcat(solutions...)
end

# Testovací data
x_min = [-5.0, -5.0]
x_max = [5.0, 5.0]
xs_init = rand(2, 10) .* 10 .- 5  # Náhodné počáteční body v intervalu [-5, 5]

# Spuštění generátoru řešení
#solutions = generate_solutions(f_griewank,g_griewank(0.0), P, x_min, x_max, xs_init)
#println("Nalezená lokální minima:")
#println(solutions)


asd = g_griewank(0.0)
@show asd