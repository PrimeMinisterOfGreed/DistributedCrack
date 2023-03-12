#include "EventHandler.hpp"
#include "Nodes/Node.hpp"
#include "OptionsBag.hpp"
#include <StringGenerator.hpp>
#include <boost/mpi.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cstdio>
#include <functional>
#include <gtest/gtest.h>
#include <md5.hpp>
#include <string>
#include <utility>


TEST(TestHandler, test_event_handler)
{
    EventHandler<> handler{};
    handler += new FunctionHandler([]() {});
}