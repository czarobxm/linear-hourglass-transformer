# 8 layers
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,4x512,2x1024 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,4x512,2x1024 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,4x512,2x1024 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,4x512,2x1024 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,4x512,2x1024 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,4x512,2x1024 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,4x512,2x1024 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,4x512,2x1024 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_8x1024 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_8x1024 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_8x1024 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_8x1024 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_8x1024 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_8x1024 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_8x1024 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_8x1024 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_8x1024 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_8x1024 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_8x1024 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_8x1024 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_8x1024 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_8x1024 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_8x1024 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_8x1024 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,4x512,2x1024 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,4x512,2x1024 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,4x512,2x1024 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,4x512,2x1024 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,4x512,2x1024 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,4x512,2x1024 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,4x512,2x1024 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,4x512,2x1024 training.optimizer.lr=0.002

# 6 layers
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,2x512,2x1024 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,2x512,2x1024 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,2x512,2x1024 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,2x512,2x1024 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,2x512,2x1024 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,2x512,2x1024 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,2x512,2x1024 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_2x1024,2x512,2x1024 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_1x1024,4x512,1x1024 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_1x1024,4x512,1x1024 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_1x1024,4x512,1x1024 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_1x1024,4x512,1x1024 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_1x1024,4x512,1x1024 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_1x1024,4x512,1x1024 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_1x1024,4x512,1x1024 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_1x1024,4x512,1x1024 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_6x1024 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_6x1024 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_6x1024 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_6x1024 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_6x1024 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_6x1024 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_6x1024 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_cosformer_6x1024 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_6x1024 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_6x1024 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_6x1024 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_6x1024 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_6x1024 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_6x1024 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_6x1024 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_6x1024 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,2x512,2x1024 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,2x512,2x1024 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,2x512,2x1024 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,2x512,2x1024 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,2x512,2x1024 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,2x512,2x1024 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,2x512,2x1024 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_2x1024,2x512,2x1024 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_1x1024,4x512,1x1024 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_1x1024,4x512,1x1024 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_1x1024,4x512,1x1024 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_1x1024,4x512,1x1024 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_1x1024,4x512,1x1024 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_1x1024,4x512,1x1024 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_1x1024,4x512,1x1024 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=pathfinder_vanilla_1x1024,4x512,1x1024 training.optimizer.lr=0.002
