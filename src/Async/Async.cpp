#include "Async/Async.hpp"

AsyncMTLoop::AsyncMTLoop(int iter, std::function<void(int)> fnc)
    : _fnc(), _iter(iter) {}
