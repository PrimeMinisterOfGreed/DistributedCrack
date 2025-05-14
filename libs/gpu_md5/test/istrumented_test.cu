#include <future>
#include <string>
#include <optional>
#include "md5_gpu.h"

int main(){

    auto start = 78074000u;
    auto end = 78074900u;
    auto target = "952bccf9afe8e4c04306f70f7bed6610";
    auto res = md5_bruter(start,end ,target, 10000,4);
    printf("result: %s\n",res.data);
}