

#include <memory>
#include <optional>
class BasePromise {
protected:
  std::optional<void *> result;
  void operator()();

public:
};

template <typename T> class Promise {}
