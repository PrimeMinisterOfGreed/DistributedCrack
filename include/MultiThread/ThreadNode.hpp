#pragma once
#include "Nodes/Node.hpp"
#include "Statistics/EventProcessor.hpp"
#include "Statistics/TimeMachine.hpp"
#include "StringGenerator.hpp"
#include <string>
#include <vector>

class ThreadMultiSchema;
class ThreadNode : public NodeHasher
{
  private:
    ThreadMultiSchema & _schema;

  public:
    ThreadNode(ThreadMultiSchema& schema, std::string target);
    void Initialize() override;
    void Routine() override;
    void OnBeginRoutine() override;
    void OnEndRoutine() override;
};