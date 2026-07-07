# Adaptive k=1 Monte Carlo Pilot Validation

## Run count
250

## partition_any counts/percentages
```
               count  percentage
partition_any                   
0                 88        35.2
1                162        64.8
```

## partition_fraction describe()
```
count    250.000000
mean       0.618667
std        0.476730
min        0.000000
25%        0.000000
50%        1.000000
75%        1.000000
max        1.000000
```

## gcc_frac_min describe()
```
count    250.000000
mean       0.506464
std        0.352636
min        0.055556
25%        0.150000
50%        0.416667
75%        0.861111
max        1.000000
```

## gcc_frac_mean describe()
```
count    250.000000
mean       0.522290
std        0.349425
min        0.066667
25%        0.171362
50%        0.448148
75%        0.866667
max        1.000000
```

## node_failure_prob describe()
```
count    250.000000
mean       0.117593
std        0.066482
min        0.002417
25%        0.062083
50%        0.116684
75%        0.164342
max        0.249356
```

## edge_failure_prob describe()
```
count    250.000000
mean       0.128879
std        0.074328
min        0.001625
25%        0.065763
50%        0.134063
75%        0.188204
max        0.249384
```

## total_satellites describe()
```
count    250.000000
mean      37.508000
std       10.688447
min       20.000000
25%       30.000000
50%       36.000000
75%       45.000000
max       60.000000
```

## node_failure_bins partition_any crosstab normalized by index
```
partition_any            0         1
node_failure_bin                    
(-0.001, 0.05]    0.500000  0.500000
(0.05, 0.1]       0.344262  0.655738
(0.1, 0.15]       0.421053  0.578947
(0.15, 0.2]       0.209302  0.790698
(0.2, 0.25]       0.230769  0.769231
```

## edge_failure_bins partition_any crosstab normalized by index
```
partition_any            0         1
edge_failure_bin                    
(-0.001, 0.05]    0.404255  0.595745
(0.05, 0.1]       0.408163  0.591837
(0.1, 0.15]       0.422222  0.577778
(0.15, 0.2]       0.232143  0.767857
(0.2, 0.25]       0.320755  0.679245
```

## total_satellite_bins partition_any crosstab normalized by index
```
partition_any               0         1
total_satellite_bin                    
(-0.001, 30.0]       0.102564  0.897436
(30.0, 45.0]         0.435897  0.564103
(45.0, 60.0]         0.527273  0.472727
```

## Comparison
```
          policy  num_runs  partition_any_positive_rate  partition_fraction_mean  partition_fraction_median  gcc_frac_min_mean  gcc_frac_min_median  gcc_frac_mean_mean  gcc_frac_mean_median  median_isolated_nodes  median_components  runtime_seconds
      grid_fixed       250                        0.784                 0.761333                        1.0           0.378266             0.176389             0.39545              0.193889                    NaN               11.0              NaN
grid_adaptive_k1       250                        0.648                 0.618667                        1.0           0.506464             0.416667             0.52229              0.448148                    NaN                6.0         2.823685
```

## Feature correlations
```
          policy           feature  pearson_partition_any  spearman_partition_any  pearson_gcc_frac_min  spearman_gcc_frac_min
      grid_fixed  total_satellites              -0.306693               -0.312102              0.471963               0.406561
      grid_fixed node_failure_prob               0.203976                0.207150             -0.099543              -0.091633
      grid_fixed edge_failure_prob               0.145817                0.142500             -0.087141              -0.074068
grid_adaptive_k1  total_satellites              -0.345702               -0.361916              0.478299               0.411901
grid_adaptive_k1 node_failure_prob               0.204333                0.206922             -0.101044              -0.183536
grid_adaptive_k1 edge_failure_prob               0.094685                0.091566             -0.060496              -0.079188
```

