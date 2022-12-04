#include "Nodes/Node.hpp"
#include <StringGenerator.hpp>
#include <boost/mpi.hpp>
#include <gtest/gtest.h>
#include <md5.hpp>

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
    }

    NodeProcessMock(std::string target) : MPINode(*new boost::mpi::communicator(), target)
    {
        _logger = new ConsoleLogEngine();
    }
};

TEST(TestNode, test_processor_node)
{
    NodeProcessMock mock{md5("%%%%")};
    mock.Execute();
}

TEST(TestNode, test_event_serialization)
{
}