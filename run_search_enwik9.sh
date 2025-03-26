sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/cosformer_2x4096,4x2048,2x4096 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/cosformer_8x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/cosformer_8x4096 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/cosformer_8x4096 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/cosformer_8x4096 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/cosformer_8x4096 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/vanilla_8x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/vanilla_8x4096 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/vanilla_8x4096 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/vanilla_8x4096 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/vanilla_8x4096 training.optimizer.lr=0.002

sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/vanilla_2x4096,4x2048,2x4096 training.optimizer.lr=0.0001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/vanilla_2x4096,4x2048,2x4096 training.optimizer.lr=0.0002
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/vanilla_2x4096,4x2048,2x4096 training.optimizer.lr=0.0005
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/vanilla_2x4096,4x2048,2x4096 training.optimizer.lr=0.001
sbatch --partition=common --qos=czarekg_common --gres=gpu:1 --time=1-0 poetry run python3 train_single_gpu.py --config-name=enwik9/vanilla_2x4096,4x2048,2x4096 training.optimizer.lr=0.002
