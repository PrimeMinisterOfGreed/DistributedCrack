#include "Async/Async.hpp"
#include "Async/Executor.hpp"
#include "CompileMacro.hpp"
#include "LogEngine.hpp"
#include "OptionsBag.hpp"
#include "StringGenerator.hpp"
#include "md5.hpp"
#include "md5_gpu.hpp"
#include <boost/mpi/communicator.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mpi.h>
#include <ostream>
#include <sstream>
#include <string>

boost::program_options::variables_map optionsMap;

void async_main() {}

int main(int argc, char *argv[]) {
  using namespace boost::program_options;
  using namespace std;
  options_description desc("allowed options: those are parsed automatically if "
                           "file dcrack.config exist");
  desc.add_options()("help", "display help message")(
      "config", value<std::string>(), "manually specify a config file")(
      "gpu", "use gpu to decrypt")("mt", "use multithread to crack")(
      "target", value<std::string>(), "the target of the crack")(
      "mpiexec", "launch the program as an mpiexec program (should use MPIexec "
                 "and then run this program)")(
      "thread", value<int>(),
      "if mt is specified indicates the number of threads to use")(
      "chunk", value<int>(),
      "specified a starting point or a fixed number of chunk for the "
      "generators")("dynamic_chunks", value<bool>(),
                    "specify if use or not dynamic chunking")(
      "verbosity", value<int>(), "specify verbosity for logger")(
      "schema", value<int>(), "specify a certain schema to use with MPI")(
      "stat", value<std::string>(),
      "save a csv file with stat from the program")(
      "restore", value<std::string>()->default_value("state.state"),
      "restore a previously saved state (default file state.state)")(
      "savefile", value<std::string>()->default_value("state.state"),
      "Indicates the file to use in case of SIGINT capture in order to save "
      "the state");
  variables_map map;
  store(parse_command_line(argc, argv, desc), map);
  if (map.contains("config") || filesystem::exists("dcrack.config")) {
    std::string filename;
    if (map.contains("config"))
      std::string filename = map.at("config").as<std::string>();
    else
      filename = "dcrack.config";
    map.clear();
    store(parse_config_file(filename.c_str(), desc), map);
  }
  notify(map);
  optionsMap = map;
  if (map.count("help")) {
    cout << desc << endl;
    cout << "Version:" << VERSION << endl;
    return 1;
  }
  if (!map.count("target")) {
    cout << "You need to specify a target before continuing (an md5 hash)"
         << std::endl;
    return 0;
  }
  auto pftr = ::async(async_main);
  *pftr += [](auto... ns) { Scheduler::main().stop(); };
  Scheduler::main().start(false);
}
