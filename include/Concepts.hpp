#pragma once
#include <concepts>
#include <functional>
#include <future>
#include <string>
#include <type_traits>
#include <vector>

template <typename T>
concept HashFunction = requires(T a, std::string input) {
                           {
                               a(input)
                               } -> std::convertible_to<std::string>;
                       };

template <typename T>
concept ComputeFunction = requires(T a, std::vector<std::string> chunk, std::string target, std::string *result,
                                   std::function<std::string(std::string)> hashFnc) {
                              {
                                  a(chunk, target, result, hashFnc)
                                  } -> std::convertible_to<bool>;
                          };

template <typename T, typename ComputeFnc>
concept ComputeAsyncFunction =
    requires(T a, std::vector<std::string> chunk, std::string target, std::string *result,
             std::function<std::string(std::string)> hashFnc) {
        {
            a(chunk, target, result, hashFnc)
            } -> std::convertible_to<std::future<bool>>;
    } &&
    ComputeFunction<ComputeFnc>;

template <typename FncPtr, typename... Args>
concept Handler = requires(FncPtr fnc, Args... args) {
                      fnc(args...);
                  };