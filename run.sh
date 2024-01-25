nohup python3 baseline_cutmix.py --alpha=0.7 --gpu=0 > /dev/null &
nohup python3 baseline_cutmix.py --alpha=0.5 --gpu=1 > /dev/null &
nohup python3 baseline_cutmix.py --alpha=0.3 --gpu=2 > /dev/null &
nohup python3 baseline_cutmix.py --alpha=0.9 --gpu=3 > /dev/null &
nohup python3 baseline_mixup.py --alpha=0.7 --gpu=4 > /dev/null &
nohup python3 baseline_mixup.py --alpha=0.5 --gpu=5 > /dev/null &
nohup python3 baseline_mixup.py --alpha=0.3 --gpu=6 > /dev/null &
nohup python3 baseline_mixup.py --alpha=0.9 --gpu=7 > /dev/null &
