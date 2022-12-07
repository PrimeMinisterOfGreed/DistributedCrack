#include "OptionsBag.hpp"
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <gtest/gtest.h>
#include <sstream>

boost::program_options::variables_map optionsMap = *new boost::program_options::variables_map{};

void fillOptionsMap()
{
    using namespace boost::program_options;
    std::stringstream options{};
    options << "stat=test.csv" << std::endl;

    options_description desc("allowed options");
    desc.add_options()("help", "display help message")("gpu", "use gpu to decrypt")("mt", "use multithread to crack")(
        "target", value<std::string>(), "the target of the crack")(
        "mpiexec", "launch the program as an mpiexec program (should use MPIexec and then run this program)")(
        "thread", value<int>(), "if mt is specified indicates the number of threads to use")(
        "chunk", value<int>(), "specified a starting point or a fixed number of chunk for the generators")(
        "dynamic_chunks", value<bool>(), "specify if use or not dynamic chunking")(
        "verbosity", value<int>(), "specify verbosity for logger")("schema", value<int>(),
                                                                   "specify a certain schema to use with MPI")(
        "stat", value<std::string>(), "save a csv file with stat from the program");

    store(parse_config_file(options, desc), optionsMap);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    fillOptionsMap();
    return RUN_ALL_TESTS();
}