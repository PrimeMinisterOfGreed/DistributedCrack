#pragma once
#include "Schema.hpp"

class ThreadNode;
class IThreadSchema : public ISchema
{
    public:
      virtual std::vector<std::string> &RequireNextSequence(ThreadNode *requiringNode) = 0;
      virtual void SignalEnd(std::string& result) = 0;
};