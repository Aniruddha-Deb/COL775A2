pip install -e .
cd nerv
pip install -e .

# create (or symlink) data to this current directory, in the following format:
# data
# `- CLEVRER
#    `- annotations
#       `- train
#          `- annotation_00000-01000
#             ...
#       `- val
#          `- annotation_10000-11000
#             ...
#       `- test
#    `- questions
#       `- train.json
#       `- val.json
#       `- test.json
#    `- videos
#       `- train
#          `- video_00000-01000
#             ...
#       `- val
#          `- video_10000-11000
#             ...
#       `- test
#          `- video_15000-16000
#             ...

# train SAVi
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 scripts/train.py --task base_slots --params slotformer/base_slots/configs/stosavi_clevrer_params.py --fp16 --ddp --cudnn

# extract slots
python slotformer/base_slots/extract_slots.py --params slotformer/base_slots/configs/stosavi_clevrer_params.py --weight checkpoint/stosavi_clevrer_params/models/model_146067.pth --save_path data/CLEVRER/slots.pkl

# train slotformer
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 scripts/train.py --task video_prediction --params slotformer/video_prediction/configs/slotformer_clevrer_params.py --fp16 --ddp --cudnn

# rollout slots
python slotformer/video_prediction/rollout_clevrer_slots.py --params slotformer/video_prediction/configs/slotformer_clevrer_params.py --weight checkpoint/slotformer_clevrer_params/models/model_18309.pth --save_path data/CLEVRER/rollout_slots.pkl

# train Aloe on rolled out slots
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 scripts/train.py --task clevrer_vqa --params slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py --fp16 --ddp --cudnn

# Evaluate aloe
python slotformer/clevrer_vqa/test_clevrer_vqa.py --params slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py --weight checkpoint/aloe_clevrer_params-rollout/models/model_238000.pth
