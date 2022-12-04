#include "LogEngine.hpp"
#include "OptionsBag.hpp"
#include "RingSchema.hpp"
#include "Schema.hpp"
#include "StringGenerator.hpp"
#include "md5.hpp"
#include <Statistics/TimeMachine.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>

boost::program_options::variables_map optionsMap;

void RunMPI(int argc, char *argv[])
{
    using namespace boost::mpi;
    MPI_Init(&argc, &argv);
    int verbosity = 0;
    if (optionsMap.count("verbosity"))
    {
        verbosity = optionsMap.at("verbosity").as<int>();
    }
    auto &comm = *new communicator();
    if (comm.rank() != 0)
    {
        std::stringstream *logBuf = new std::stringstream();
        MPILogEngine::CreateInstance(comm, nullptr, &std::cout, verbosity);
    }
    else
    {
        MPILogEngine::CreateInstance(comm, nullptr, &std::cout, verbosity);
    }
    MPILogEngine::Instance()->TraceInformation("Starting process:{0}", comm.rank());
    int schema = 0;
    std::string target = optionsMap.at("target").as<std::string>();
    if (optionsMap.count("schema"))
        schema = optionsMap.at("schema").as<int>();
    int chunk = 2000;
    if (optionsMap.count("chunk"))
        chunk = optionsMap.at("chunk").as<int>();
    switch (schema)
    {
    case 0:
        SimpleMasterWorker(chunk, target).ExecuteSchema(comm);
        break;

    case 1:
        MasterWorkerDistributedGenerator(chunk, target).ExecuteSchema(comm);
        break;
    }
    MPI_Finalize();
}

int main(int argc, char *argv[])
{
    using namespace boost::program_options;
    using namespace std;
    options_description desc("allowed options");
    desc.add_options()("help", "display help message")("gpu", "use gpu to decrypt")("mt", "use multithread to crack")(
        "target", value<std::string>(), "the target of the crack")(
        "mpiexec", "launch the program as an mpiexec program (should use MPIexec and then run this program)")(
        "thread", value<int>(),
        "if mt is specified indicates the number of threads to use else indicates the number of MPI process")(
        "chunk", value<int>(), "specified a starting point or a fixed number of chunk for the generators")(
        "dynamic_chunks", value<bool>(), "specify if use or not dynamic chunking")(
        "verbosity", value<int>(), "specify verbosity for logger")("target", value<int>(), "target md5 to crack")(
        "schema", value<int>(), "specify a certain schema to use with MPI");
    variables_map map;
    store(parse_command_line(argc, argv, desc), map);
    notify(map);
    optionsMap = map;
    if (map.count("help"))
    {
        cout << desc << endl;
        return 1;
    }
    if (!map.count("target"))
    {
        cout << "You need to specify a target before continuing (an md5 hash)" << std::endl;
        return 0;
    }
    if (map.count("mpiexec"))
    {
        RunMPI(argc, argv);
        return 0;
    }
}