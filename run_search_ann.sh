# 8 layers
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_8x4096 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_8x4096 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_8x4096 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_8x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_8x4096 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_8x4096 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_8x4096 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_8x4096 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_8x4096 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_8x4096 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_8x4096 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_8x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_8x4096 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_8x4096 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_8x4096 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_8x4096 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,4x2048,2x4096 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,4x2048,2x4096 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,4x2048,2x4096 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,4x2048,2x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,4x2048,2x4096 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,4x2048,2x4096 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,4x2048,2x4096 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,4x2048,2x4096 training.optimizer.lr=0.002

# 6 layers
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,2x2048,2x4096 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,2x2048,2x4096 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,2x2048,2x4096 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,2x2048,2x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,2x2048,2x4096 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,2x2048,2x4096 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,2x2048,2x4096 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_2x4096,2x2048,2x4096 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_1x4096,4x2048,1x4096 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_1x4096,4x2048,1x4096 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_1x4096,4x2048,1x4096 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_1x4096,4x2048,1x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_1x4096,4x2048,1x4096 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_1x4096,4x2048,1x4096 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_1x4096,4x2048,1x4096 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_1x4096,4x2048,1x4096 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_6x4096 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_6x4096 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_6x4096 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_6x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_6x4096 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_6x4096 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_6x4096 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_cosformer_6x4096 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_6x4096 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_6x4096 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_6x4096 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_6x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_6x4096 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_6x4096 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_6x4096 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_6x4096 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,2x2048,2x4096 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,2x2048,2x4096 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,2x2048,2x4096 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,2x2048,2x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,2x2048,2x4096 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,2x2048,2x4096 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,2x2048,2x4096 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_2x4096,2x2048,2x4096 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_1x4096,4x2048,1x4096 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_1x4096,4x2048,1x4096 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_1x4096,4x2048,1x4096 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_1x4096,4x2048,1x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_1x4096,4x2048,1x4096 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_1x4096,4x2048,1x4096 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_1x4096,4x2048,1x4096 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=ann_vanilla_1x4096,4x2048,1x4096 training.optimizer.lr=0.002
