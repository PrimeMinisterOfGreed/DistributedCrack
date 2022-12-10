#include "Nodes/Node.hpp"
#include "OptionsBag.hpp"
#include <StringGenerator.hpp>
#include <boost/mpi.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cstdio>
#include <gtest/gtest.h>
#include <md5.hpp>
#include <string>
#include <utility>

class NodeComputeMock : public MPINode
{
  public:
    // Ereditato tramite MPINode
    virtual void Routine() override
    {
        SequentialGenerator generator{4};
        std::string result;
        while (!Compute(generator.generateChunk(2000), &result))
        {
        }
    }
    virtual void Initialize() override
    {
    }
    virtual void OnBeginRoutine() override
    {
    }
    virtual void OnEndRoutine() override
    {
    }

    NodeComputeMock(std::string target) : MPINode(*new boost::mpi::communicator(), target)
    {
        _logger = new ConsoleLogEngine();
    }

    void SetTarget(std::string target)
    {
        _target = target;
    }
};

TEST(testNode, testNodeCompute)
{
    NodeComputeMock mock{md5("!!!!")};
    mock.Execute();
}

class NodeProcessMock : public MPINode
{
  private:
  public:
    // Ereditato tramite MPINode
    virtual void Routine() override
    {
        SequentialGenerator generator{4};
        std::string result;
        while (!Compute(generator.generateChunk(2000), &result))
        {
        }
    }

    virtual void Initialize() override
    {
    }

    virtual void OnBeginRoutine() override
    {
        _stopWatch.Start();
    }

    virtual void OnEndRoutine() override
    {
        auto stat = _processor.ComputeStatistics();
        std::cout << stat.ToString() << std::endl;
        AddResult(stat, 0, "Test");
    }

    NodeProcessMock(std::string target) : MPINode(*new boost::mpi::communicator(), target)
    {
        _logger = new ConsoleLogEngine();
    }
};

TEST(TestNode, test_processor_node)
{
    auto value = std::pair<std::string, std::string>("stat", "test.csv");
    NodeProcessMock mock{md5("!!%!")};
    mock.Execute();
}
