#include "clock_register.hpp"

using namespace std::chrono;

std::pair<void*, time_point<system_clock>> _clocks[128]{};

system_clock global_clock{};

time_point<system_clock> start_point;


uint64_t ClockRegister::clock_since_start() {
    auto now = global_clock.now();
    auto res = duration_cast<milliseconds>(now-start_point).count();
    return res;
}

void ClockRegister::tick(void* ctx) {
    for(int i = 0 ;i< sizeof(_clocks)/sizeof(_clocks[0]); i++){
        if(_clocks[i].first == nullptr){
            _clocks[i].first = ctx;
            _clocks[i].second = global_clock.now();
        }
    }
}

uint64_t ClockRegister::tock(void* ctx) {
        for(int i = 0 ;i< sizeof(_clocks)/sizeof(_clocks[0]); i++){
        if(_clocks[i].first == ctx){
            _clocks[i].first = nullptr;
            return duration_cast<milliseconds>(global_clock.now() - _clocks[i].second).count();
        }
    }
    perror("Clock doesn't exist");
    exit(-1);
}

void ClockRegister::init() {
    
    start_point = global_clock.now();                
}


