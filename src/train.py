from transformers import (
    get_linear_schedule_with_warmup, 
    get_cosine_with_min_lr_schedule_with_warmup_lr_rate,
    get_constant_schedule_with_warmup
)

def get_scheduler(optimizer, hparams, num_train_steps, cycles=1):
    warmup_steps = int(num_train_steps * hparams.get("warmup_ratio", 0))
    
    stype = hparams["scheduler_type"]
    
    if stype == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=num_train_steps
        )
    elif stype == "cosine":
        return get_cosine_with_min_lr_schedule_with_warmup_lr_rate(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=num_train_steps,
            num_cycles=cycles
        )
    elif stype == "constant":
        return get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {stype}")