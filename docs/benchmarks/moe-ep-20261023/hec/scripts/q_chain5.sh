cd $HOME/moe-ep/hec
. $HOME/moe-ep/hec/env_hec.sh
$RUN $LOGD build_chain5 ./buildplug.sh      # under the bench lock
bash q_s10.sh
bash q_s8.sh
bash q_gt430.sh
bash q_lfm.sh
