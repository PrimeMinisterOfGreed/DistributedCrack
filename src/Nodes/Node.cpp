#include "Nodes/Node.hpp"
#include "DataContainer.hpp"
#include "Functions.hpp"
#include "OptionsBag.hpp"
#include "md5.hpp"
#include <future>
#include <string>
#include <thread>

void Node::AddResult(Statistics &statistic, int process, std::string method)
{
    ExecutionResult result{};
    result.process = process;
    result.method = method;
    result.statistics = statistic;
    _container->AddResult(result);
}

void Node::Execute()
{
    try
    {
        Initialize();
        BeginRoutine();
        Routine();
        EndRoutine();
    }
    catch (const std::exception &ex)
    {
        _logger->TraceException(ex.what());
        _logger->Finalize();
    }
}

void Node::BeginRoutine()
{
    _logger->TraceInformation("Routine Setup");
    try
    {
        OnBeginRoutine();
    }
    catch (const std::exception &e)
    {
        _logger->TraceException("Exception during routine setup:{0}", e.what());
        throw;
    }
    _logger->TraceInformation("Routine Setup completed");
}

void Node::EndRoutine()
{
    _logger->TraceInformation("Ending Routine");
    try
    {
        OnEndRoutine();
    }
    catch (const std::exception &e)
    {
        _logger->TraceException("Exception during routine ending:{0}", e.what());
        throw;
    }
    _logger->TraceInformation("Routine end done, saving results if any");
    if (_container->HasDataToSave() && optionsMap.contains("stat"))
    {
        _container->SaveToFile(optionsMap.at("stat").as<std::string>().c_str());
    }
}

void Node::ExecuteRoutine()
{
    _logger->TraceInformation("Routine Execution");
    try
    {
        Routine();
    }
    catch (const std::exception &e)
    {
        _logger->TraceException("Exception during routine execution:{0} ", e.what());
        throw;
    }
    _logger->TraceInformation("Routine execution done");
}

void MPINode::DeleteRequest(boost::mpi::request &request)
{
    int index = indexOf<boost::mpi::request>(_requests.begin(), _requests.end(),
                                             [&](boost::mpi::request val) -> bool { return &val == &request; });
    if (index == -1)
        throw std::invalid_argument("Index of request is not existent");
    _requests.erase(_requests.begin() + index);
}

bool NodeHasher::Compute(const std::vector<std::string> &chunk, std::string *result,
                         std::function<std::string(std::string)> hashFnc)
{
    bool comp = false;
    auto ev = _stopWatch.RecordEvent([&](Event &e) {
        size_t completions = 0;
        for (auto &string : chunk)
        {
            completions++;
            if (hashFnc(string) == _target)
            {
                _logger->TraceInformation("Founded password: ", string);
                *result = string;
                comp = true;
                e.completitions = completions;
                break;
            }
        }
        e.completitions = completions;
    });
    _processor.AddEvent(ev);
    return comp;
}

std::future<bool> NodeHasher::ComputeAsync(const std::vector<std::string> &chunk,
                                           std::function<void(std::string)> callback)
{
    return std::async([&]() -> bool {
        std::string result = "";
        if (Compute(chunk, &result))
        {
            callback(result);
            return true;
        }
        else
        {
            callback("NULL");
            return false;
        }
    });
}

Statistics &NodeHasher::GetNodeStats() const
{
    return _processor.ComputeStatistics();
}
