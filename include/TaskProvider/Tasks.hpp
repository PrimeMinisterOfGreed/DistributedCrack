#pragma once


#include <string>
#include <vector>
class ITask
{
  private:
    bool _used = false;

  public:
    inline bool used() const
    {
        return _used;
    }
    inline void setUsed()
    {
        _used = true;
    }
};

