C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
GGML_SGLX_LAZY_COPY=0 ./kld.sh pp_C128_nolazy "$C" cpu
GGML_SGLX_LAZY_COPY=0 KB=512 KUB=320 ./kld.sh pp_C320_nolazy "$C" cpu
