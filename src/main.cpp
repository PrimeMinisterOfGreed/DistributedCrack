#include "RingSchema.hpp"
#include "Schema.hpp"
#include "StringGenerator.hpp"
#include "md5.hpp"
#include "OptionsBag.hpp"
#include <TimeMachine.hpp>
#include <boost/mpi/communicator.hpp>
#include "LogEngine.hpp"
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <sstream>

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
        std::stringstream* logBuf = new std::stringstream();
        MPILogEngine::CreateInstance(comm, logBuf, logBuf,verbosity);
    }
    else
    {
        MPILogEngine::CreateInstance(comm, nullptr, &std::cout,verbosity);
    }
    auto time = executeTimeComparison([&]() {
        SimpleMasterWorker schema{1000, *new std::string("0000")};
        schema.ExecuteSchema(comm);
    });
    if (comm.rank() == 0)
        std::cout << "Executed in time: " << time.count() << "ms" << std::endl;
    MPI_Finalize();
}

int main(int argc, char *argv[])
{
    using namespace boost::program_options;
    using namespace std;
    options_description desc("allowed options");
    desc.add_options()("help", "display help message")("mpi", "use mpi to decrypt")(
        "gpu", value<bool>(), "use gpu to decrypt")("mt", value<bool>(), "use multithread to crack")(
        "target", value<std::string>(), "the target of the crack")(
        "mpiexec", value<bool>(), "launch the program as an mpiexec program [use mpi if command line]")(
        "thread", value<int>(),
        "if mt is specified indicates the number of threads to use else indicates the number of MPI process")(
        "chunk", value<int>(), "specified a starting point or a fixed number of chunk for the generators")(
        "dynamic_chunks", value<bool>(), "specify if use or not dynamic chunking")
            ("verbosity",value<int>(),"specify verbosity for logger");

    variables_map map;
    store(parse_command_line(argc, argv, desc), map);
    notify(map);
    int verbosity = 0;
    if (map.count("verbosity"))
    {
        verbosity = map.at("verbosity").as<int>();
    }
    optionsMap = map;
    if (map.count("help"))
    {
        cout << desc << endl;
        return 1;
    }
    else if (map.count("mpi"))
    {
        stringstream command;
        int maxProc = 4;
        if (map.count("thread"))
            maxProc = map["thread"].as<int>();
        command << "mpiexec -np " << maxProc << " " << argv[0] << " --mpiexec on --verbosity " << verbosity;
        system(command.str().c_str());
    }
    else if (map.count("mpiexec"))
    {
        RunMPI(argc, argv);
    }
}