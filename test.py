module load anaconda/2024.02-py311
python3 -c "
import zarr
z = zarr.open('/anvil/scratch/x-jhong6/data/surya_bench_train_hour_8.zarr', mode='r')
print('Keys:', list(z.keys()))
for key in z.keys():
    print(f'  {key}: shape={z[key].shape}, dtype={z[key].dtype}')
"