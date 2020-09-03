module MinkReduction

using LinearAlgebra, Printf
export GaussReduce, RandUnimodMat, RandLowerTri, minkReduce, DeviousMat, isMinkReduced

"""
    minkReduce(U, V, W, debug=false)

Find the shortest equivalent basis of that lattice formed by {`U`, `V`, `W`}

```jldoctest
julia> U = [1, 2, 3]; V = [-1, 2, 3]; W = [3, 0, 4]; minkReduce(U,V,W)
[1.0 2.0 -1.0; 2.0 -2.0 2.0; 3.0 1.0 3.0]
([1.0, 2.0, 3.0], [2.0, -2.0, 1.0], [-2.0, 0.0, 0.0])
```
"""
function minkReduce(U, V, W; debug=false)
    i = 0
    origMat = hcat(U,V,W)
    println("Input basis vectors:")
    println("U: ",U, "  norm: ",round(norm(U),digits=2))
    println("V: ",V, "  norm: ",round(norm(V),digits=2)) 
    println("W: ",W, "  norm: ",round(norm(W),digits=2))
    while true
        i +=1
        println(" --< iteration $i >--")
        norms = [norm(U), norm(V), norm(W)]
        println("sort vectors into ascending order by ℓ-2 norms")
        p = sortperm(norms)
        U,V,W = (U,V,W)[p] # sort into ascending order
        U,V,W = shortenW_in_UVW(U, V, W)
        #println("integer transform? ", inv(origMat)*hcat(U,V,W))
        i > 10 && error("minkReduce: Too many iterations") 
        norm(W) ≥ norm(V) && break
    end
    println("final vectors:")
    if !debug return U, V, W end
    return U, V, W, i
end

"""
    shortenW_in_UVW

Reduce vector W so that it is as close to the origin as possible.

Gauss reduce U and V. Subtract multiples of new U, V from W to bring its 
U-V projection as close as possible to the origin. The projection of the new
W will be contained in the parallelogram formed by U, V. 
(See Lecture notes in computer science, ISSN 0302-974, ANTS - VI : algorithmic 
number theory, 2004, vol. 3076, pp. 338-357 ISBN 3-540-22156-5)
"""
function shortenW_in_UVW(U,V,W)
    U, V = GaussReduce(U,V)
    println("Next: Shorten W in U-V-W")
    println("U:",U)
    println("V:",V)
    println("W:",W)
    @printf("norms: U, V, W:\n %7.3f\n %7.3f\n %7.3f\n ", norm(U), norm(V), norm(W))
    # n̂ is a unit vector ⟂ to U-V plane. Subtract W⋅n̂ multiples of n̂
    # from W to get W's projection, T, into the U-V plane. 
    n̂ = (U×V)/norm(U×V)
    T = W - W⋅n̂ * n̂ # Get the projection of W into U-V plane
    𝕄 = hcat(U,V,W) # Matrix with U, V, W as columns
    # Get the number of multiples of U, V needed to move T inside the U-V cell    
    println("unrounded/unfloored multiples of U,V,W: ", inv(𝕄)*T)
    latCoords = floor.(Int,inv(𝕄)*T) 
     println("U,V multiples needed to shift W into first cell: ",latCoords)
     println("Shift in Cartesian coords: ", 𝕄*latCoords)
    # Find the corner of the U-V cell closest to shifted T and shift T to be closest to origin
    Wnew = W - 𝕄*latCoords # Try the corner at the origin first
        # Now try the other three corners, keep the shortest that is found
    norm(Wnew) > norm(W - U) && (Wnew = W - U)
    norm(Wnew) > norm(W - V) && (Wnew = W - V)
    norm(Wnew) > norm(W - U - V) && (Wnew = W - U -V)
    W = Wnew
    println("shortened W:",W)
    @printf("norms: U, V, Wnew:\n %7.3f\n %7.3f\n %7.3f\n ", norm(U), norm(V), norm(Wnew))
    return U, V, Wnew
end

"""
    GaussReduce(U, V)

Reduce the basis vectors {`U`, `V`} to the shortest possible basis.

# Examples
```julia-repl
julia> GaussReduce([5 8], [8 13])
([-1.0 0.0], [0.0 -1.0])
```
"""
function GaussReduce(U, V)
    println("GaussReduce begins")
    #maxval = max(abs.(U)...,abs.(V)...)
    if norm(U) > norm(V) U, V = V, U end
    i = 0
    while true
        println("U: ",U,"  V: ",V)
        V, U = U, V - round(Int,(U⋅V)/(U⋅U))*U
        i += 1
        if norm(U) > norm(V) || norm(U)≈norm(V) break; end
        i > 50 && error("GaussReduce: Too many iterations") # failsafe to break out if not converging
    end
    println("U: ",U,"  V: ",V)
    println("GaussReduce complete")
    return V, U
end

"""
    RandUnimodMat(n)

Generate a random unimodular 2x2 matrix. `n` is a small integer (number of row and column operations).

See also: `RandLowerTri(n)`, `FibonacciMat(n)`, `DeviousMat(n)`

# Examples
```jldoctest
julia> RandUnimodMat(5)
2×2 Array{Int64,2}:
 3   8
 7  19
```
"""
function RandUnimodMat(n)
    mat = RandLowerTri(1)
    for i ∈ 1:n
        mat = mat*RandLowerTri(1)
        mat = mat*transpose(RandLowerTri(1))
    end
    return mat
end

"""
    RandLowerTri(n)
Generate a random 2x2 matrix of the form [1 0; 0 ±n].

See also: `RandUnimodMat(n)`, `FibonacciMat(n)`, `DeviousMat(n)`

# Examples
```jldoctest
julia> RandLowerTri(4)
2×2 Array{Int64,2}:
  1  0
 -3  1
```
"""
function RandLowerTri(n)
    return [1 0; rand(-n:n) 1]
end

"""
    FibonacciMat(k)
Generate a 2x2 matrix of the form [f2 f3; f1 f2] where f1, f2, f3 consecutive Fibonacci-like

See also: `RandUnimodMat(n)`, `FibonacciMat(n)`, `DeviousMat(n)`

"""
function FibonacciMat(k)
    f1 = round(Int32,1.61803398875^k/sqrt(5))
    f2 = round(Int32,1.61803398875^(k+1)/sqrt(5))
    f3 = f1 + f2
    any(i -> i < 1, [f1 f2 f3]) && error("Overflow in FibonacciMat function")
    return [f2 f3; f1 f2]
end

"""
    DeviousMat(n)

Make a unimodular 3x3 matrix that requires a large number of steps to reduce
(See email from Rod Feb 1 2020)

`n` dictates the size of the entries
"""
function DeviousMat(n)
    n < 3 && error("for DeviousMat, n ≥ 3")
    u,v = round(Int64,(2+√3)^n/(2*√3)), round(Int64,(2+√3)^n/2)
    a,b = convert(Int64,(u+v+1)/2), -u
    c,d = a-1, v-u
    return [a;b;c], [b;d;b], [c;b;a]
end

"""
    isMinkReduced(U,V,W) 

Check if the basis {`U`,`V`,`W`} is Minkoswki reduced.
    
"""
function isMinkReduced(U,V,W)
    if norm(U) > norm(V)+eps()     println("Condition 1  failed"); return false end
    if norm(V) > norm(W)+eps()     println("Condition 2  failed"); return false end
    if norm(V) > norm(U+V)+eps()   println("Condition 3  failed"); return false end
    if norm(V) > norm(U-V)+eps()   println("Condition 4  failed"); return false end
    if norm(W) > norm(U+W)+eps()   println("Condition 5  failed"); return false end
    if norm(W) > norm(U-W)+eps()   println("Condition 6  failed"); return false end
    if norm(W) > norm(V+W)+eps()   println("Condition 7  failed"); return false end
    if norm(W) > norm(V-W)+eps()   println("Condition 8  failed"); return false end
    if norm(W) > norm(U+V+W)+eps() println("Condition 9  failed"); return false end
    if norm(W) > norm(U-V+W)+eps() println("Condition 10 failed"); return false end
    if norm(W) > norm(U+V-W)+eps() println("Condition 11 failed"); return false end
    if norm(W) > norm(U-V-W)+eps() println("Condition 12 failed"); return false end
    return true
end

end

