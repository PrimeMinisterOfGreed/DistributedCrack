#pragma once
#include <concepts>
#include <functional>
#include <future>
#include <string>
#include <type_traits>
#include <vector>

template <typename Archive, typename T>
concept Serializable =
    requires(T t, Archive &a, const unsigned int v) { a.serialize(t, v); };
