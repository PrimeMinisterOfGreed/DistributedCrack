#pragma once
#include "Schema.hpp"
#include "MultiThread/ThreadNode.hpp"
#include "MultiThread/MultiThreadStringGenerator.hpp"
#include "Statistics/EventProcessor.hpp"
#include <vector>
#include <map>

class ThreadMultiSchema : public ISchema
{
    int _threads;
    MultiThreadStringGenerator& _mtGenerator;
    std::vector<ThreadNode> _nodes;
    std::string _target;
    int _initialChunkSize;
    std::map<const ThreadNode*,Statistics> _lastRunStat; 
  public:
    ThreadMultiSchema(int threads, int initialSequenceLength, std::string target, int initialChunkSize);
    void Initialize() override;
    void ExecuteSchema() override;
    std::vector<std::string>& RequireNextSequence(const ThreadNode * requiringNode);
};
