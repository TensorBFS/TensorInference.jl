# The "Asia" Bayesian network

Please check the Julia code [asia.jl](asia.jl).

The variables and factors for the asia model is described in the [asia.uai](asia.uai) file.
The UAI file format is detailed in:
https://personal.utdallas.edu/~vibhav.gogate/uai16-evaluation/uaiformat.html

The meanings of variables and factors as listed bellow.

## Variables
index 0 is mapped to yes, 1 is mapped to no.

1. visit to Asia (a)
2. tuberculosis (t)
3. smoking (s)
4. lung cancer (l)
5. bronchitis (b)
6. either tub. or lung cancer (e)
7. positive X-ray (x)
8. dyspnoea (d)

## Factors
1. p(a)
2. p(t|a)
3. p(s)
4. p(l|s)
5. p(b|s)
6. p(e|l,t)
7. p(x|e)
8. p(d|e,b)
