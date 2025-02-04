# For MIND-small
python TaFR/src/GRV/main.py \
    --dataset MIND-small \
    --path ../../../data/MIND \
    --prediction_dataset MIND-small \
    --prediction_path ../prediction/COX/COX_MIND-small_24h \
    --T_obs 12 \       # From paper
    --beta_E 0.5 \        # From paper
    --beta_nE 0.5 \
    --beta_d -3

# For full MIND dataset
python TaFR/src/GRV/main.py \
    --dataset MIND \
    --path ../../../data/MIND \
    --prediction_dataset MIND \
    --prediction_path ../prediction/COX/COX_MIND_24h \
    --T_obs 12 \
    --beta_E 0.5 \
    --beta_nE 0.5 \
    --beta_d -3