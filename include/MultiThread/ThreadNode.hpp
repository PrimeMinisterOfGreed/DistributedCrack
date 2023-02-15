#include "Nodes/Node.hpp"
#include "Statistics/EventProcessor.hpp"
#include "StringGenerator.hpp"
#include <string>
#include <vector>
#include "Statistics/TimeMachine.hpp"
class ThreadNode : public Node
{
  private:
    EventProcessor & _processor = *new EventProcessor();
    StopWatch & _stopWatch;
    std::string _target;
    ISequenceGenerator &_generator;
    bool Compute(std::vector<std::string>& chunk, std::string target, std::string * value);
  public:
    ThreadNode(ISequenceGenerator& generator, std::string target);
    void Initialize() override;
    void Routine() override;
    void OnBeginRoutine() override;
    void OnEndRoutine() override;
};