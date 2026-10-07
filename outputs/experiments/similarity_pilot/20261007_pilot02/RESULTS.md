**Phan Nguyễn Mai Phương — MSHV 25002711**

# Similarity pilot FD001

      procedure  endpoint_n_rows  endpoint_n_engines  endpoint_mean_window_mae  endpoint_mean_engine_mae  endpoint_rmse  endpoint_nasa_score  endpoint_mean_engine_nasa  trajectory_n_rows  trajectory_n_engines  trajectory_mean_window_mae  trajectory_mean_engine_mae  trajectory_rmse  trajectory_nasa_score  trajectory_mean_engine_nasa  on_time_rate  on_time_count  late_or_missed_rate  missed_warning_count  early_alarm_burden  mean_transitions  mean_retractions  engines_with_retractions  median_lead_time  min_lead_time  max_lead_time  n_engines  policy_threshold  policy_persistence
hgb_baseline169              300                 100                 10.550971                 10.550971      16.450557          6154.845143                  20.516150              17731                   100                   17.646544                   16.750309        28.835035           1.521142e+07                   530.477933          0.94             94                 0.06                     0            0.055746              1.82              0.41                        31              38.0           28.0           55.0        100                40                   1
 similarity_k10              300                 100                 20.843944                 20.843944      28.490233         31157.943621                 103.859812              17731                   100                   25.332270                   23.848124        36.543007           6.819236e+07                  2161.178136          0.92             92                 0.08                     0            0.099723              2.02              0.51                        41              43.0           26.0           84.0        100                50                   1
  similarity_k5              300                 100                 20.889846                 20.889846      30.426953        231282.477435                 770.941591              17731                   100                   25.684236                   24.612412        37.172962           9.761201e+07                  3186.061226          0.91             91                 0.09                     0            0.073286              2.88              0.94                        61              43.0           24.0           74.0        100                45                   1

Decision:

     procedure  pooled_rmse_lower  fold_rmse_wins  fold_rmse_wins_2_of_3  endpoint_mae_not_worse  endpoint_nasa_not_worse  on_time_not_worse  burden_not_worse  pilot_continue_candidate
 similarity_k5              False               0                  False                   False                    False              False             False                     False
similarity_k10              False               0                  False                   False                    False              False             False                     False

Pilot status: {"status": "completed", "elapsed_seconds": 17.038053800002672, "test_labels_read": false, "pilot_continue": false}
