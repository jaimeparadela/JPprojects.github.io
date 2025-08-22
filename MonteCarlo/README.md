Some implementations of the Monte Carlo method for option pricing:

  -European option via discretization of the GBM SDE:
  
      -Computation of option price
      -Computation of delta via two different formulas: direct definition and Malliavin weight. For T \to 0 we observe variance reduction: the direct definition presents smaller variance (see Theory file about variance reduction).
  
  -Zero-coupon bonds via: 
  
      - discretization of the sqrt diffusion SDE,
      
      -using the exact transition distribution kernel
