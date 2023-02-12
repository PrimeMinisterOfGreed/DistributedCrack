#include "Node.hpp"

class ThreadNode : public Node
{
  public:
    ThreadNode();
    void Initialize() override;
    void Routine() override;
};