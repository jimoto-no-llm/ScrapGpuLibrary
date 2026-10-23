cd $HOME/moe-ep/hec
. $HOME/moe-ep/hec/env_hec.sh
$RUN $LOGD build_chain5 ./buildplug.sh      # under the bench lock
bash q_s10.sh
bash q_kld1.sh
bash q_gt430.sh
bash q_prior.sh
bash q_nat.sh
bash q_nat2.sh
bash q_base3.sh
bash q_final.sh
bash q_lfm_moe.sh
bash q_lfm.sh
