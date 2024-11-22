
#include <argparse.hpp>
#include <mpi/mpi.h>
#include <string>
#include "Async/executor.hpp"
#include "options_bag.hpp"
#include "string_generator.hpp"
#include <log_engine.hpp>
#include<boost/mpi.hpp>
#include "MultiThread/functions.hpp"


void send_chunks(boost::mpi::communicator& comm, int chunks, int target){
  static AssignedSequenceGenerator gen{4};
  auto data = gen.generate_chunk(chunks);
  
}


void mpi_routine(int argc, char **argv) {
  using namespace boost::mpi;
  MPI_Init(&argc, &argv);
  communicator comm{};
  auto rank = comm.rank();
  char *data = NULL;
  std::vector<int> sizes{};
  


  MPI_Finalize();
}

int main(int argc, char *argv[]) {
  using namespace std;
  auto parser = argparse::ArgumentParser{"dist_crack"};
#define make_arg(argument, help_msg)                                           \
  parser.add_argument(argument).help(help_msg)
  auto options = ProgramOptions::instance();

  make_arg("--config", "load a config file")
      .store_into(options->config_file)
      .default_value("NONE"),
      make_arg("--gpu", "use_gpu")
          .flag()
          .store_into(options->use_gpu)
          .default_value(false),
      make_arg("--target", "tageted hash to decrypt")
          .required()
          .store_into(options->target_md5),
      make_arg("--threads", "number of threads to use")
          .default_value(1)
          .store_into(options->num_threads),
      make_arg("--chunk_size", "size of chunks to digest for each thread")
          .default_value(1000)
          .store_into(options->chunk_size),
      make_arg("--verbosity", "verbosity of the log file")
          .default_value(1)
          .store_into(options->verbosity),
      make_arg("--restore", "restore a previous save file")
          .flag()
          .store_into(options->restore_from_file),
      make_arg("--save_file", "store file for data")
          .default_value("store.dat")
          .store_into(options->savefile),
      make_arg("--mpi", "assume launched with mpi")
          .flag()
          .store_into(options->use_mpi),
      make_arg("--dictionary", "use a dictionary")
          .default_value("NONE")
          .store_into(options->dictionary);
  parser.parse_args(argc, argv);
  mpi_routine(argc, argv);

  return 0;
}
