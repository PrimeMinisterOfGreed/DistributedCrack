#include "Schema.hpp"
#include "MultiThread/ThreadNode.hpp"
#include "MultiThread/MultiThreadStringGenerator.hpp"


class ThreadMultiSchema : public ISchema
{
    int _threads;
    MultiThreadStringGenerator& _mtGenerator;
    std::vector<ThreadNode> _nodes;
  public:
    ThreadMultiSchema(int threads, int initialSequenceLength);
    void Initialize() override;
    void ExecuteSchema() override;
};
