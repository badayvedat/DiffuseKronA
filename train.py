import os
from datetime import datetime

from cli import get_train_args
from env import SDXL_MODEL_CACHE, SDXL_URL
from preprocess import preprocess
from trainer_pti import main
from utils import download_weights

args = get_train_args()

# Hard-code token_map for now. Make it configurable once we support multiple concepts or user-uploaded caption csv.
token_map = args.token_string + ":2"

# Process 'token_to_train' and 'input_data_tar_or_zip'
inserting_list_tokens = token_map.split(",")

token_dict = {}
running_tok_cnt = 0
all_token_lists = []
for token in inserting_list_tokens:
    n_tok = int(token.split(":")[1])

    token_dict[token.split(":")[0]] = "".join(
        [f"<s{i + running_tok_cnt}>" for i in range(n_tok)]
    )
    all_token_lists.extend([f"<s{i + running_tok_cnt}>" for i in range(n_tok)])

    running_tok_cnt += n_tok


input_dir = preprocess(
    input_images_filetype=args.input_images_filetype,
    input_zip_path=args.input_images,
    caption_text=args.caption_prefix,
    mask_target_prompts=args.mask_target_prompts,
    target_size=args.resolution,
    crop_based_on_salience=args.crop_based_on_salience,
    use_face_detection_instead=args.use_face_detection_instead,
    temp=args.clipseg_temperature,
    substitution_tokens=list(token_dict.keys()),
)

if not os.path.exists(SDXL_MODEL_CACHE):
    download_weights(SDXL_URL, SDXL_MODEL_CACHE)

# create a temp dir based on formatted current time in format of 'year-month-day-hour-minute-second'
OUTPUT_DIR = (
    args.output_dir or f"train-local-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

main(
    pretrained_model_name_or_path=SDXL_MODEL_CACHE,
    instance_data_dir=os.path.join(input_dir, "captions.csv"),
    output_dir=OUTPUT_DIR,
    seed=args.seed,
    resolution=args.resolution,
    train_batch_size=args.train_batch_size,
    num_train_epochs=args.num_train_epochs,
    max_train_steps=args.max_train_steps,
    gradient_accumulation_steps=1,
    ti_lr=args.ti_lr,
    krona_lr=args.krona_lr,
    lr_scheduler=args.lr_scheduler,
    lr_warmup_steps=args.lr_warmup_steps,
    token_dict=token_dict,
    inserting_list_tokens=all_token_lists,
    verbose=args.verbose,
    checkpointing_steps=args.checkpointing_steps,
    allow_tf32=True,
    mixed_precision="bf16",
    device="cuda:0",
)
