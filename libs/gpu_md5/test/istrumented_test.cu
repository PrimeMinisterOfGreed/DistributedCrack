#include <future>
#include <string>
#include <optional>
#include "md5_gpu.h"

int main(){
    size_t offset=  pow(94,5);
    auto start = offset;
    auto end = offset+1000;
    auto target = "5d41402abc4b2a76b9719d911017c592";
    auto res = md5_bruter(start,end ,target, 1000,4);
    printf("result: %s\n",res.data);
}