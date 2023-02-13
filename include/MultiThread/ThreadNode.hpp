#include "Nodes/Node.hpp"
#include "StringGenerator.hpp"

class ThreadNode : public Node
{
  private:
    ISequenceGenerator & _generator;

  public:
    ThreadNode(ISequenceGenerator& generator);
    void Initialize() override;
    void Routine() override;
    void OnBeginRoutine() override;
    void OnEndRoutine() override;
};