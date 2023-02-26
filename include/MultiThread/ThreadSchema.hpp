#pragma once
#include "DataContainer.hpp"
#include "LogEngine.hpp"
#include "Schema.hpp"
#include "MultiThread/ThreadNode.hpp"
#include "MultiThread/MultiThreadStringGenerator.hpp"
#include "Statistics/EventProcessor.hpp"
#include "MultiThread/AutoResetEvent.hpp"
#include <vector>
#include <map>
#include <mutex>


class ThreadMultiSchema : public IThreadSchema
{
  private:
    
    int _threads;
    MultiThreadStringGenerator &_mtGenerator;
    AutoResetEvent _waitEnd{false};
    std::vector<ThreadNode> _nodes;
    DataContainer& _container = *new DataContainer();
    ILogEngine * _logger;
    std::string _target;
    std::string _result;
    int _initialChunkSize;
    std::map<const ThreadNode*,Statistics> _lastRunStat; 
  public:
    ThreadMultiSchema(int threads, int initialSequenceLength, std::string target, int initialChunkSize, ILogEngine * logger);
    void Initialize() override;
    void ExecuteSchema() override;
    std::vector<std::string> &RequireNextSequence(ThreadNode *requiringNode) override;
    void SignalEnd(std::string &result) override;
    std::string GetResult() const{return _result;}
};
