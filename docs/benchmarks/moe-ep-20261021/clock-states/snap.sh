#!/bin/bash
# snap.sh OUT: text snapshot of the RX 6400 (card1) and WX 2100 (card5) driver state (no heavy reads).
{
for c in 1 5; do d=/sys/class/drm/card$c/device; echo "##### card$c"
  for f in power_dpm_force_performance_level power_dpm_state pp_cur_state pp_num_states pp_dpm_sclk pp_dpm_mclk pp_dpm_fclk pp_dpm_socclk pp_dpm_pcie pp_power_profile_mode pp_features pp_sclk_od pp_mclk_od power/runtime_status power/runtime_active_time power/runtime_suspended_time power/runtime_usage power/control power/autosuspend_delay_ms current_link_speed current_link_width; do
    [ -r $d/$f ] && { echo "--- $f"; cat $d/$f 2>&1; }; done
  for h in $d/hwmon/hwmon*; do for f in $h/*_input $h/power1_average $h/power1_cap $h/pwm1 $h/pwm1_enable $h/fan1_target; do [ -r $f ] && echo "hwmon/$(basename $f) $(cat $f 2>&1)"; done; done
  [ -r $d/pp_table ] && echo "pp_table md5 $(md5sum < $d/pp_table)"
done
echo "##### cpu"; grep . /sys/devices/system/cpu/cpu0/cpufreq/{scaling_cur_freq,energy_performance_preference,scaling_governor}; cat /sys/devices/system/cpu/intel_pstate/{no_turbo,min_perf_pct,max_perf_pct}
} > "$1" 2>&1
