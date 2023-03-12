#pragma once
#include <concepts>
#include <functional>
#include <future>
#include <string>
#include <type_traits>
#include <vector>

template <typename Archive, typename T>
concept Serializable = requires(T t, Archive &a, const unsigned int v) { a.serialize(t, v); };
template <typename T>
concept HashFunction = requires(T a, std::string input) {
                           {
                               a(input)
                               } -> std::convertible_to<std::string>;
                       };

template <typename T, typename Hasher>
concept ComputeFunction = requires(T a, std::vector<std::string> chunk, std::string target, std::string *result,
                                   Hasher hashFnc) {
                              {
                                  a(chunk, target, result, hashFnc)
                                  } -> std::convertible_to<bool>;
                          } && HashFunction<Hasher>;

template <typename T, typename Hasher>
concept ComputeAsyncFunction =
    requires(T a, std::vector<std::string> chunk, std::string target, std::string *result,
             Hasher hashFnc) {
        {
            a(chunk, target, result, hashFnc)
            } -> std::convertible_to<std::future<bool>>;
    } &&
    HashFunction<Hasher>;

template <typename FncPtr, typename... Args>
concept Handler = requires(FncPtr fnc, Args... args) {
                      fnc(args...);
                      {fnc == fnc}-> std::convertible_to<bool>;
                      };

