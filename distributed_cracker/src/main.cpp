
#include <argparse.hpp>
#include <string>

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

int main(int argc, char *argv[]) {
  using namespace std;
  auto parser = argparse::ArgumentParser{"dist_crack"};
  #define make_arg(argument,help_msg) parser.add_argument(argument).help(help_msg)

  make_arg("--help", "display this help").flag().store_into(use_gpu);
  make_arg("--config", "load a config file").store_into(config_file);
  make_arg("--gpu", "use_gpu").flag().store_into(use_gpu).default_value(false);
  make_arg("--target", "tageted hash to decrypt").required().store_into(target_md5);
  make_arg("--threads", "number of threads to use").default_value(1).store_into(num_threads);
  make_arg("--chunk_size", "size of chunks to digest for each thread").default_value(1000).store_into(chunk_size);
  make_arg("--verbosity", "verbosity of the log file").default_value(1).store_into(verbosity);
  make_arg("--restore", "restore a previous save file").flag().store_into(restore_from_file);
  make_arg("--save_file", "store file for data").default_value("store.dat").store_into(savefile);
  make_arg("--mpi", "assume launched with mpi").flag().store_into(use_mpi);
  make_arg("--dictionary", "use a dictionary").default_value("NONE").store_into(dictionary);

  
}
