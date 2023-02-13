#include "Schema.hpp"
#include "Nodes/ThreadNode.hpp"

class ThreadMultiSchema : public ISchema
{
    int _threads;
    std::vector<ThreadNode> _nodes;
  public:
    ThreadMultiSchema(int threads);
    void Initialize() override;
    void ExecuteSchema() override;
};
