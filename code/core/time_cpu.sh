for nodes in 32 64 128 256 512 1024 2048 4096 8192
do
    export OMP_NUM_THREADS=4
    python3 timing_hmc.py --cpu -p "(1000022, 1000022)" --epochs 0 --batch 32 --burn 100 --kernel hmc --results 10 --chains 1 --skip 10 --arch "[5, ${nodes}, 1]" --act "tanh" --leapfrog 512
done
