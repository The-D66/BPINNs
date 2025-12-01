import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, width_ratios=[15, 3, 1], wspace=0.05, hspace=0.3)


def calc_variance_trace(data, t_arr, window_sec=100.0):
  dt_avg = (t_arr[-1] - t_arr[0]) / (len(t_arr) - 1)
  window_steps = int(window_sec / dt_avg)
  if window_steps < 1:
    window_steps = 1

  vars = []
  ts = []

  for i in range(0, len(t_arr) - window_steps, window_steps):
    chunk = data[i:i + window_steps, :]
    var_val = np.mean(np.var(chunk, axis=0))
    vars.append(var_val)
    ts.append(t_arr[i])

  return np.array(vars), np.array(ts)


ax_h = fig.add_subplot(gs[0, 0])
ax_h_var = fig.add_subplot(gs[0, 1], sharey=ax_h)
ax_h_cb = fig.add_subplot(gs[0, 2])

im_h = ax_h.imshow(
    h_history, aspect='auto', extent=[0, L, T_total_sim, 0], cmap='viridis'
)
ax_h.axhline(y=T_warmup, color='k', linestyle='--', label='Warmup End')
ax_h.set_ylabel('t (s)')
ax_h.set_xlabel('x (m)')
ax_h.set_title('Water Depth (Full Simulation)')
ax_h.legend(loc='upper right')

plt.colorbar(im_h, cax=ax_h_cb, label='Water Depth h (m)')

h_vars, h_ts = calc_variance_trace(h_history, t_history, window_sec=100.0)
ax_h_var.plot(h_vars, h_ts, 'k-', linewidth=1)
ax_h_var.fill_betweenx(h_ts, 0, h_vars, color='gray', alpha=0.5)
ax_h_var.axhline(y=T_warmup, color='k', linestyle='--')
ax_h_var.set_xlabel('Variance (100s)')
ax_h_var.set_xscale('log')

h_v_max = np.max(h_vars)
h_v_min = h_v_max / 1000.0
ax_h_var.set_xlim(h_v_min, h_v_max * 1.2)

plt.setp(ax_h_var.get_yticklabels(), visible=False)
ax_h_var.invert_yaxis()
ax_h_var.grid(True, axis='x', alpha=0.3)

ax_q = fig.add_subplot(gs[1, 0])
ax_q_var = fig.add_subplot(gs[1, 1], sharey=ax_q)
ax_q_cb = fig.add_subplot(gs[1, 2])

im_q = ax_q.imshow(
    Q_history, aspect='auto', extent=[0, L, T_total_sim, 0], cmap='viridis'
)
ax_q.axhline(y=T_warmup, color='k', linestyle='--', label='Warmup End')
ax_q.set_ylabel('t (s)')
ax_q.set_xlabel('x (m)')
ax_q.set_title('Discharge (Full Simulation)')
ax_q.legend(loc='upper right')

plt.colorbar(im_q, cax=ax_q_cb, label='Discharge Q (m³/s)')

q_vars, q_ts = calc_variance_trace(Q_history, t_history, window_sec=100.0)
ax_q_var.plot(q_vars, q_ts, 'k-', linewidth=1)
ax_q_var.fill_betweenx(q_ts, 0, q_vars, color='gray', alpha=0.5)
ax_q_var.axhline(y=T_warmup, color='k', linestyle='--')
ax_q_var.set_xlabel('Variance (100s)')
ax_q_var.set_xscale('log')

q_v_max = np.max(q_vars)
q_v_min = q_v_max / 1000.0
ax_q_var.set_xlim(q_v_min, q_v_max * 1.2)

plt.setp(ax_q_var.get_yticklabels(), visible=False)
ax_q_var.invert_yaxis()
ax_q_var.grid(True, axis='x', alpha=0.3)

plt.savefig("data_raw/reference_solution_full.png", bbox_inches='tight')
print("Reference plot saved to data_raw/reference_solution_full.png")
