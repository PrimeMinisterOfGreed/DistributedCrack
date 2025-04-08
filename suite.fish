set LD_PRELOAD_PATH "/home/drfaust/Scrivania/uni/Magistrale/SCPD/Project/DistributedCrack/build/libs/gpu_md5/libmd5_gpu.so"

function cracker_test -a testname
    LD_PRELOAD=$LD_PRELOAD_PATH cargo test --package cracker --lib  -- $testname  --show-output
end

function cracker_mpi_test -a testname
    LD_PRELOAD=$LD_PRELOAD_PATH mpiexec -n 2  cargo test --package cracker --lib  -- $testname  --show-output --nocapture
end