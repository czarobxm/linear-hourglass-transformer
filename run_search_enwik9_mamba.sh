# sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_8 training.optimizer.lr=0.0001
# sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_8 training.optimizer.lr=0.0002
# sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_8 training.optimizer.lr=0.0005
# sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_8 training.optimizer.lr=0.001
# sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_8 training.optimizer.lr=0.002


# sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_2x4x2 training.optimizer.lr=0.0001
# sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_2x4x2 training.optimizer.lr=0.0002
# sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_2x4x2 training.optimizer.lr=0.0005
# sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_2x4x2 training.optimizer.lr=0.001
# sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_2x4x2 training.optimizer.lr=0.002

sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_6 training.optimizer.lr=0.0001
sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_6 training.optimizer.lr=0.0002
sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_6 training.optimizer.lr=0.0005
sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_6 training.optimizer.lr=0.001
sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_6 training.optimizer.lr=0.002

sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_1x4x1 training.optimizer.lr=0.0001
sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_1x4x1 training.optimizer.lr=0.0002
sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_1x4x1 training.optimizer.lr=0.0005
sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_1x4x1 training.optimizer.lr=0.001
sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_1x4x1 training.optimizer.lr=0.002

sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_2x2x2 training.optimizer.lr=0.0001
sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_2x2x2 training.optimizer.lr=0.0002
sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_2x2x2 training.optimizer.lr=0.0005
sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_2x2x2 training.optimizer.lr=0.001
sbatch --gres=gpu:1 --cpus-per-task=1 --ntasks=1 --mem=120G --time=2-0 --partition=plgrid-gpu-a100 --account=plgllmparamgr-gpu-a100 poetry run python3 train_single_gpu.py --config-name=enwik9_mamba_2x2x2 training.optimizer.lr=0.002