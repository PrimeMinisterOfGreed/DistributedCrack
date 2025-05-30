#pragma once
#include "options.hpp"
#include <cstdint>
#include <vector>
#include <cstring>
#include <chrono>
struct TimerStats{
    uint64_t busy_time = 0;
    uint64_t observation_time = 0;
    uint64_t task_completed = 0;
    static char device_name[32];
    char name[32]{};
    TimerStats(const char * name);
    std::string to_csv();
    static std::string csv_header();
    static void set_device_name(const char * name);
};

struct TimerContext{
    private:
    std::chrono::high_resolution_clock::time_point start_time;
    TimerStats& stats;
    public:

    void start();
    void stop();
    TimerContext(const char* name);
    ~TimerContext();
    
    
    template<typename Fnc>
    void with_context(Fnc fnc)
    {
        start();
        fnc(stats);
        stop();
    }
    
    

};

struct GlobalClock{
    private:
    std::vector<TimerStats> stats;
    std::chrono::high_resolution_clock::time_point start_time;
    char device_name[32]{};
    public:
    static GlobalClock& instance();
    TimerStats& get_or_create(const char * name);
    const decltype(start_time) get_time() const
    {
        return start_time;
    }

    decltype(stats) get_stats() const
    {
        return stats;
    }
    private:
    GlobalClock();

};



void save_stats(const char* filename);
void init_stat_file(const char* filename);