#include "ToString.hpp"
#include "LinkedList.hpp"
#include <string>



template <> std::string EventList::ToString()
{
    std::stringstream buffer{};
    auto ptr = begin();
    while (ptr != end() && &ptr() != nullptr)
    {
        buffer << "|Job:" << (*ptr).Name << ",TA:" << std::to_string((*ptr).ArrivalTime)
               << " ,TS:" << std::to_string((*ptr).ServiceTime) << ", Type:" << (char)(*ptr).Type
               << ", TO:" << std::to_string((*ptr).OccurTime) << "|-->";
        ptr++;
    }
    return *new std::string(buffer.str());
}