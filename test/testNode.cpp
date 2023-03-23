#include "Compute.hpp"
#include "Concepts.hpp"
#include "EventHandler.hpp"
#include "LogEngine.hpp"
#include "Nodes/Node.hpp"
#include "OptionsBag.hpp"
#include <StringGenerator.hpp>
#include <boost/mpi.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <gtest/gtest.h>
#include <md5.hpp>
#include <string>
#include <utility>
#include "TaskProvider/HashTask.hpp"
#include "TaskProvider/TaskGenerator.hpp"
#include "TaskProvider/TaskRunner.hpp"

TEST(TestHandler, test_event_handler)
{
    EventHandler<> handler{};
    handler += new FunctionHandler([]() {});
}


bool NullPredicate(HashTask& task){return task.result != nullptr;}

TEST(TestRunner, test_local_runner)
{
    auto engine = new ConsoleLogEngine(3);
    auto targetHash = md5("%%%");
    HashTaskGenerator gen(2000, targetHash ,4);
    TaskRunner<HashTask, HashTaskGenerator> tasker(gen, [](HashTask &task) { return task.result != nullptr; });
    auto &node = *new HashNode(Compute(md5), tasker, tasker, engine);
    tasker.Execute();
    node.Execute();
}