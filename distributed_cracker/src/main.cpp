
#include <argparse.hpp>
#include <mpi/mpi.h>
#include <string>
#include "Actors/root.hpp"
#include "Actors/worker.hpp"
#include "options_bag.hpp"
#include <log_engine.hpp>
#include <boost/mpi.hpp>


ProgramOptions options{};


void single_node_routine(){

}

void mpi_routine(int argc, char **argv) {
  using namespace boost::mpi;
  MPI_Init(&argc, &argv);
  communicator comm{};
  auto rank = comm.rank();
  if (rank == 0) {
    println("Master PID:{}",getpid());
    root_routine(comm);
  } else {
    println("Process {}, pid {}",rank,getpid());
    worker_routine(comm);
  }

  MPI_Finalize();
}

int main(int argc, char *argv[]) {
  using namespace std;
  auto parser = argparse::ArgumentParser{"dist_crack"};
#define make_arg(argument, help_msg)                                           \
  parser.add_argument(argument).help(help_msg)

  make_arg("--config", "load a config file")
      .store_into(options.config_file)
      .default_value("NONE"),
      make_arg("--gpu", "use_gpu")
          .flag()
          .store_into(options.use_gpu)
          .default_value(false),
      make_arg("--target", "tageted hash to decrypt")
          .required()
          .store_into(options.target_md5),
      make_arg("--threads", "number of threads to use")
          .default_value(1)
          .store_into(options.num_threads),
      make_arg("--chunk-size", "size of chunks to digest for each thread")
          .default_value(1000)
          .store_into(options.chunk_size),
      make_arg("--verbosity", "verbosity of the log file")
          .default_value(1)
          .store_into(options.verbosity),
      make_arg("--restore", "restore a previous save file")
          .flag()
          .store_into(options.restore_from_file),
      make_arg("--save_file", "store file for data")
          .default_value("store.dat")
          .store_into(options.savefile),
      make_arg("--mpi", "assume launched with mpi")
          .flag()
          .store_into(options.use_mpi),
      make_arg("--dictionary", "use a dictionary")
          .default_value("NONE")
          .store_into(options.dictionary);
          make_arg("--brutestart", "start from a string of length n for bruteforce")
          .default_value(4)
          .store_into(options.brutestart);
  parser.parse_args(argc, argv);
  if(options.use_mpi)
    mpi_routine(argc, argv);
  else
    single_node_routine();
  return 0;
}
