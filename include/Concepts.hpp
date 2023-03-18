#pragma once
#include <concepts>
#include <functional>
#include <future>
#include <string>
#include <type_traits>
#include <vector>

template <typename Archive, typename T>
concept Serializable = requires(T t, Archive &a, const unsigned int v) { a.serialize(t, v); };

template <typename T, typename Ret ,typename... Args>
concept Callable = requires(T a, Args... args) {
                       {a(args...)}->std::convertible_to<Ret>;
};



template <typename T>
concept HashFunction = requires(T a, std::string input) {
                           {
                               a(input)
                               } -> std::convertible_to<std::string>;
                       };

template <typename T>
concept ComputeFunction = requires(T a, std::vector<std::string> chunk, std::string target, std::string *result) {
                              {
                                  a(chunk, target, result)
                                  } -> std::convertible_to<bool>;
                          };

template <typename T>
concept ComputeAsyncFunction =
    requires(T a, std::vector<std::string> chunk, std::string target, std::string *result) {
        {
            a(chunk, target, result)
            } -> std::convertible_to<std::future<bool>>;
    };

template <typename FncPtr, typename... Args>
concept Handler = requires(FncPtr fnc, Args... args) {
                      fnc(args...);
                      {fnc == fnc}-> std::convertible_to<bool>;
                      };


template <typename Task, typename FncGen>
concept TaskGenerator = requires(FncGen a) {
                            {
                                a()
                                } -> std::convertible_to<Task>;
                        };


class BaseComputeNode;

template <typename Task, typename Provider>
concept TaskProvider = requires(Provider p, BaseComputeNode &node, Task &task) {
                           p.RequestTask(&node);
                           p.CheckResult(task);
                       };


template <typename SignalProvider>
concept NodeSignaler = requires(SignalProvider provider, BaseComputeNode& node) {
                           provider.RegisterNode(node);
};
