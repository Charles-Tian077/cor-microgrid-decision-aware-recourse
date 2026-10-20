# Active model and algorithm

## Settlement and decisions

The inherited day-ahead purchase `q0[t]` remains payable. At a non-midnight release, the revised energy is `q[t] = q0[t] + up[t] - down[t]`. The primary objective is the sum of predicted prices multiplied by `1.5*up + 0.5*down + 5*emergency`. The inherited charge is constant for the optimizer but is included in realized totals.

Variables are revised purchase, charging input, delivered discharge, emergency purchase, unused surplus, stored energy and directional adjustments. All energy variables are kWh. The supply inequality is implemented with nonnegative surplus:

```text
q + discharge + emergency - charge - surplus = net_target
soc_next = soc + 0.9*charge - discharge/0.9
1200 <= soc <= 10800
0 <= charge, discharge <= 5000/6
0 <= downward adjustment <= inherited purchase
```

Eliminating surplus yields supply at least as large as target demand. A supply-equality implementation without surplus would be a different model. Downward adjustment carries an additional charge, not a refund.

## Lexicographic selection

Both stages use `scipy.optimize.linprog(method='highs')`. First minimize financial recourse. Then minimize `sum(charge+discharge) + 1e-6*sum(surplus)` subject to primary cost no larger than its optimum plus `max(1e-5, 1e-10*abs(primary optimum))`. This is a secondary tie-breaker, not a battery-degradation charge. The stage time limit is 60 seconds and primal/dual feasibility tolerances are `1e-7` in the active experiment wrapper.

With H remaining periods there are `8H+1` variables, `3H+1` equality rows and `12H+1` equality-matrix nonzeros. The second stage adds one cost-lock inequality. It does not add integer variables. Under free spill and round-trip losses, state-preserving cancellation justifies removal of simultaneous charging and discharge; the conclusion does not automatically extend to export-revenue or positive-spill-cost models.

## Release / execution boundary

The 00:00–06:00 segment uses the inherited schedule. At 06:00, 12:00 and 18:00, solve for the remaining day and execute only the next segment. Realized storage initializes the next window and continues across midnight. A gain blends optimized with inherited commands; principal comparisons use 0, 0.5 and 1. Each current-period action is physically corrected using current observed net demand and measured storage, with emergency energy covering any residual shortage.

Grid and emergency suppliers are not capacity-limited. LP feasibility under this assumption is different from resilience to a grid outage. A forecast-conditioned LP optimum is not a proof that the entire online annual policy is globally optimal.

## Learning and selection

The statistical candidates are histogram gradient boosting, a matched-history boosting model and a three-seed GRU ensemble. Predictors are fitted using statistical losses. The operational screen then evaluates fusion weights 0, 0.25, 0.5 and 1 by chronological validation settlement cost. It is not end-to-end differentiation through the optimizer. All evaluated families select zero learned-price fusion for the archived comparison.

The active implementation is `code/original/expanded/cor_q4_experiment.py` calling `q4_core.solve_window`. `q4_core.py` also retains unused upstream routines; their presence does not make CVaR, joint scenarios or terminal rewards part of the active receding-horizon controller.
