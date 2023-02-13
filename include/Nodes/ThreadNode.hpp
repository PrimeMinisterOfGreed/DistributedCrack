#include "Node.hpp"

class ThreadNode : public Node
{
  public:
    void Initialize() override;
    void Routine() override;
    void OnBeginRoutine() override;
    void OnEndRoutine() override;
};