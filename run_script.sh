sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.0002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_cosformer_8x4096 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_cosformer_8x4096 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_cosformer_8x4096 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_cosformer_8x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_cosformer_8x4096 training.optimizer.lr=0.0002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_vanilla_8x4096 training.optimizer.lr=0.00001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_vanilla_8x4096 training.optimizer.lr=0.00002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_vanilla_8x4096 training.optimizer.lr=0.00005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_vanilla_8x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9_vanilla_8x4096 training.optimizer.lr=0.0002