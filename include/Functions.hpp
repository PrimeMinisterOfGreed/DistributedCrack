#pragma once
#include <vector>

template <typename T>
int indexOf(typename std::vector<T>::iterator begin, typename std::vector<T>::iterator end,
    std::function<bool(T&)> predicate)
{
    auto b = begin;
    int i = 0;
    while (b != end && !predicate(*b))
    {
        i++;
        b++;
    }
    return b == end ? -1 : i;
}
