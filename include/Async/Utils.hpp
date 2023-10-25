#pragma once
#include "Executor.hpp"
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <vector>

void waitAny(std::vector<boost::intrusive_ptr<Task>> tasks);
void waitAll(std::vector<boost::intrusive_ptr<Task>> tasks);
