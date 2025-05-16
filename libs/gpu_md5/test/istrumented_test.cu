#include <future>
#include <string>
#include <optional>
#include "md5_gpu.h"

int main(){

    auto start = pow(94,5)-100;
    auto end = pow(94,5)+100;
    auto target = "952bccf9afe8e4c04306f70f7bed6610";
    auto res = md5_bruter(start,end ,target, 100,1);
    printf("result: %s\n",res.data);
}