set LD_PRELOAD_PATH "(pwd)/build/libs/gpu_md5/libmd5_gpu.so"

function cracker_test -a testname
    LD_PRELOAD=$LD_PRELOAD_PATH cargo test --package cracker --lib  -- $testname  --show-output
end

function cracker_mpi_test -a testname
    LD_PRELOAD=$LD_PRELOAD_PATH mpiexec -n 2  cargo test --package cracker --lib  -- $testname  --show-output --nocapture
end

function mpc_gtest -a testpattern
    LD_PRELOAD=$LD_PRELOAD_PATH mpiexec -n 2 (pwd)/build/core/test/mpc_ut --gtest_filter="$testpattern" --gtest_brief=0
end

function gtest $argv
    LD_PRELOAD=$LD_PRELOAD_PATH ./build/core/test/mpc_ut  $argv
end

function run -a process $argv
    LD_PRELOAD=$LD_PRELOAD_PATH mpirun -n $process ./build/core/mpc  $argv
end