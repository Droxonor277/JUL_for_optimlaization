using LinearAlgebra
using Statistics
using RDatasets

function computeQ(X, y)
    """Compute the Q matrix for the dual SVM problem."""
    n = size(X, 1)
    Q = zeros(n, n)
    for i in 1:n
        for j in 1:n
            Q[i, j] = y[i] * y[j] * dot(X[i, :], X[j, :])
        end
    end
    return Q
end

function computeW(X, y, z)
    """Compute the primal solution w from the dual solution z."""
    return sum((y .* z) .* X, dims=1)[:]
end

function solve_SVM_dual(Q, C; kwargs...)
    """Solve the dual SVM problem using coordinate descent."""
    n = size(Q, 1)
    z = zeros(n)
    for epoch in 1:get(kwargs, :max_epoch, 100)
        for i in 1:n
            grad = dot(Q[i, :], z) - 1
            d = Q[i, i] != 0 ? -grad / Q[i, i] : 0
            z[i] = clamp(z[i] + d, 0, C)
        end
    end
    return z
end

function solve_SVM(X, y, C; kwargs...)
    """Solve the full SVM problem by computing Q, solving the dual, and computing w."""
    Q = computeQ(X, y)
    z = solve_SVM_dual(Q, C; max_epoch=get(kwargs, :max_epoch, 100))
    w = computeW(X, y, z)
    return w
end

function iris(C; kwargs...)
    """Load the Iris dataset, preprocess it, and compute the optimal separating hyperplane w."""
    df = dataset("datasets", "iris") |> DataFrame
    X = df[:, [:PetalLength, :PetalWidth]] |> Matrix  # PetalLength and PetalWidth
    y = ifelse.(df[:, :Species] .== "versicolor", 1, -1)
    
    # Select only versicolor (1) and virginica (2) classes
    mask = (df[:, :Species] .== "versicolor") .| (df[:, :Species] .== "virginica")
    X, y = X[mask, :], y[mask]
    y = ifelse.(y .== 1, 1, -1)  # Convert labels to {1, -1}
    
    # Normalize features
    X = (X .- mean(X, dims=1)) ./ std(X, dims=1)
    
    # Add bias term
    X = hcat(X, ones(size(X, 1)))
    
    # Train SVM
    w = solve_SVM(X, y, C; kwargs...)
    return w
end