#include "Schema.hpp"
#include "MultiThread/ThreadNode.hpp"
#include "MultiThread/MultiThreadStringGenerator.hpp"


class ThreadMultiSchema : public ISchema
{
    int _threads;
    MultiThreadStringGenerator& _mtGenerator;
    std::vector<ThreadNode> _nodes;
    std::string _target;
  public:
    ThreadMultiSchema(int threads, int initialSequenceLength, std::string target);
    void Initialize() override;
    void ExecuteSchema() override;
};
