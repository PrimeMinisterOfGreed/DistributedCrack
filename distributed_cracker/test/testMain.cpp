#include <gtest/gtest.h>

std::string config_file;
bool use_gpu;
std::string target_md5;
int num_threads;
int chunk_size;
int verbosity;
std::string savefile;
bool ismpi;
bool restore_from_file;
bool use_mpi;
std::string dictionary;

void fillOptionsMap()
{

}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

