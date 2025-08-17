sbatch --nodes=1 --ntasks=1 --gres=gpu:1 --time=01:00:00 poetry run python test_time_infer.py --config-name=copying_cosformer_6x128
sbatch --nodes=1 --ntasks=1 --gres=gpu:1 --time=01:00:00 poetry run python test_time_infer.py --config-name=copying_cosformer_2x128,2x64,2x128
sbatch --nodes=1 --ntasks=1 --gres=gpu:1 --time=01:00:00 poetry run python test_time_infer.py --config-name=copying_cosformer_1x128,4x64,1x128

sbatch --nodes=1 --ntasks=1 --gres=gpu:1 --time=01:00:00 poetry run python test_time_infer.py --config-name=copying_cosformer_8x128
sbatch --nodes=1 --ntasks=1 --gres=gpu:1 --time=01:00:00 poetry run python test_time_infer.py --config-name=copying_cosformer_2x128,4x64,2x128
