#include "Concepts.hpp"
#include <gtest/gtest.h>
#include <vector>
#include "md5.hpp"
#include "Compute.hpp"

void Fun(int, int)
{
    
}

template <typename Function>
    requires Callable<Function, void, int, int>
class FunMock
{
    public:FunMock(Function fun){}
};

template <typename HFun>
    requires HashFunction<HFun>
class FunHashMock 
{
      public: FunHashMock( HFun fun){}
};

template <ComputeFunction CFun>
class FunCompMock
{
        public: FunCompMock(CFun fun){}
};

TEST(TestConcepts, test_compile)
{
          using vs = std::vector<std::string>;
          using s = std::string;
        FunMock fun(Fun);
        FunHashMock f2(md5);
       FunCompMock(Compute(md5));
        
}
