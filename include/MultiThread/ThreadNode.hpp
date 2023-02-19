#pragma once
#include "LogEngine.hpp"
#include "MultiThread/IThreadSchema.hpp"
#include "Nodes/Node.hpp"
#include "Statistics/EventProcessor.hpp"
#include "Statistics/TimeMachine.hpp"
#include "StringGenerator.hpp"
#include <string>
#include <vector>

class ThreadNode : public NodeHasher
{
  private:
    IThreadSchema *_schema;
    bool _end = false;

  protected:
    void Initialize() override;
    void Routine() override;
    void OnBeginRoutine() override;
    void OnEndRoutine() override;
  public:
    ThreadNode(IThreadSchema *schema, std::string target, ILogEngine* logger);
    virtual void Execute() override;
    void ForceEnd()
    {
        _end = true;
    };
};