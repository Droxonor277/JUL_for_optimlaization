abstract type BracketingMethod end

struct Bisection <: BracketingMethod end
struct RegulaFalsi <: BracketingMethod end

function midpoint(::Bisection, f, a, b)
    """ 
    Returns the newly calculated point using 
    the bisection method 
    """
    return (a + b) / 2

end

function midpoint(::RegulaFalsi, f, a, b)
    """ 
    Returns the newly calculated point using 
    the regula falsi method
    """
    fa, fb = f(a), f(b)
    return (a * fb - b * fa) / (fb - fa)
end

function findroot(
    method::BracketingMethod,
    f::Function,
    a::Real,
    b::Real; 
    atol::Real=1e-8,
    maxiter::Int=1000
)
    """
    The function returns the newly found root  
    within the allowed tolerance atol after the 
    specified maximum number of iterations iter
    """
    if a > b
        a, b = b, a
    end

    fa, fb = f(a), f(b)

    if fa * fb > 0
        throw(DomainError("Funkční hodnoty v bodech a a b 
                            musí mít opačné znaménko."))
    end

    # If one of the two points is within the required tolerance,
    # return it
    if abs(fa) < atol
        return a
    elseif abs(fb) < atol
        return b
    end

    # Calculation of the roots using the chosen method 
    for i in 1:maxiter
        c = midpoint(method, f, a, b)
        fc = f(c)

        if abs(fc) < atol
            return c
        end

        if fa * fc < 0
            b, fb = c, fc
        else
            a, fa = c, fc
        end
    end

    throw(ErrorException("Maximální počet iterací 
                        dosažen bez nalezení kořene."))
end