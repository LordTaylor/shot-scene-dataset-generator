# LoRA Training Guide

Once you've generated a dataset with this tool, use it to train a LoRA with Kohya or any compatible trainer.

---

## Recommended Dataset Size

| Purpose | Images |
|---------|--------|
| Quick test | 20–40 |
| Decent LoRA | 60–100 |
| High quality | 150–300 |

Use **varied presets** across runs so poses, shots, and scenes don't repeat:

```bash
python3 gen_dataset.py --char my_char --preset full    --count 40
python3 gen_dataset.py --char my_char --preset action  --count 30
python3 gen_dataset.py --char my_char --preset minimal --count 20
```

---

## Captioning

After generation, caption each image with WD14 or Florence-2.  
Keep the trigger word consistent across all captions:

```
my_char_trigger, standing, hands on hips, golden hour sunlight, ...
```

Remove any tags that describe the style of the base model — you want the LoRA to learn the *character*, not the style.

---

## Kohya Training (simplified)

```bash
# 1. Install kohya_ss
git clone https://github.com/bmaltais/kohya_ss

# 2. Prepare dataset directory
dataset/
  my_char/
    images/
      100_my_char_trigger/    # repetition_number_trigger
        image_001.png
        image_001.txt         # caption file
        ...

# 3. Run training
python train_network.py \
  --pretrained_model_name_or_path /path/to/base_model.safetensors \
  --train_data_dir dataset/my_char/images \
  --output_dir output/my_char_lora \
  --network_module networks.lora \
  --network_dim 32 \
  --network_alpha 16 \
  --learning_rate 1e-4 \
  --max_train_steps 2000 \
  --save_every_n_steps 250
```

---

## Testing Your LoRA in ComfyUI

1. Move the `.safetensors` to your ComfyUI `models/loras/` folder
2. Add `LoraLoader` node between your checkpoint and KSampler
3. Set `lora_name` to your file, `strength_model` = 0.7, `strength_clip` = 0.7
4. Include the trigger word in your positive prompt
5. Run a few test generations at different seeds

---

## Tips

- **More variety = better generalization.** Use `--preset chaos` for maximum diversity.
- **Avoid duplicates.** Set different `--seed` ranges for each run.
- **Batch size 1** is safer than 2 for most setups (avoids VRAM-related artifacts).
- **cfg 7** works well for Illustrious-based checkpoints; lower (4–5) for SDXL-based.
