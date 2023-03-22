#pragma once
#include <concepts>
#include <functional>
#include <future>
#include <string>
#include <type_traits>
#include <vector>

template <typename Archive, typename T>
concept Serializable = requires(T t, Archive &a, const unsigned int v) { a.serialize(t, v); };

template <typename T, typename Ret, typename... Args>
concept Callable = requires(T a, Args... args) {
                       {
                           a(args...)
                           } -> std::convertible_to<Ret>;
                   };

template <typename T>
concept HashFunction = Callable<T, std::string, std::string>;

template <typename T>
concept ComputeFunction = requires(T a, std::vector<std::string> chunk, std::string target, std::string *result) {
                              {
                                  a(chunk, target, result)
                                  } -> std::convertible_to<bool>;
                          };

template <typename T>
concept ComputeAsyncFunction = requires(T a, std::vector<std::string> chunk, std::string target, std::string *result) {
                                   {
                                       a(chunk, target, result)
                                       } -> std::convertible_to<std::future<bool>>;
                               };

template <typename FncPtr, typename... Args>
concept Handler = requires(FncPtr fnc, Args... args) {
                      fnc(args...);
                      {
                          fnc == fnc
                          } -> std::convertible_to<bool>;
                  };

template <typename Task, typename FncGen>
concept TaskGenerator = requires(FncGen generator) {
                            {
                                generator()
                                } -> std::convertible_to<Task>;
                        };

template <typename Node, typename Task, typename Provider>
concept TaskProvider = requires(Provider p, Node &node, Task &task) {
                           p.RequestTask(node);
                           //    p.CheckResult(task);
                       };

template <typename Node, typename Task> class ITaskProvider
{
  public:
    virtual void RequestTask(Node &node) = 0;
    virtual void CheckResult(Task &task) = 0;
};

template <typename Node, typename SignalProvider>
concept NodeSignaler = requires(SignalProvider provider, Node &node) { provider.RegisterNode(node); };

template <typename Node> class ISignalProvider
{
  public:
    virtual void RegisterNode(Node &node) = 0;
};

template <typename Task, typename Obj>
concept ComputeObject = requires(Obj computeObject, Task &task) {
                            computeObject.EnqueueTask(task);
                            computeObject.Abort();
                        };

template <typename Task> class IComputeObject
{
  public:
    virtual void Enqueue(Task &task) = 0;
    virtual void Abort() = 0;
};

