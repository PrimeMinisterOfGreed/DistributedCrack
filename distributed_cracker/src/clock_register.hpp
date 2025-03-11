#pragma once
#include <chrono>

struct ClockRegister{
    static uint64_t clock_since_start();
    static void tick(void* ctx);
    static uint64_t tock(void* ctx);
    static void init();
};