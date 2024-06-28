CUDA_VISIBLE_DEVICES=0,  python launch.py \
    --config configs/multi-prompt_benchmark/asd_sd_hyper_iNGP_50k.yaml \
    --test \
    system.prompt_processor.prompt_library="magic3d_15_prompt_library" \
    system.weights="pretrained/3d_checkpoints/MG15_hyper_iNGP.pth"

# # better to run in multi-gpu mode
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_sd_hyper_iNGP_50k.yaml \
#     --test \
#     system.prompt_processor.prompt_library="magic3d_15_prompt_library" \
#     system.weights="pretrained/3d_checkpoints/MG15_hyper_iNGP.pth"