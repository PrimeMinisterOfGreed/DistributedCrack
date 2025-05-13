#pragma once
#include <cstdint>
#include <vector>
#include <cstring>
#include <chrono>
struct TimerStats{
    uint64_t busy_time;
    uint64_t observation_time;
    uint64_t task_completed;
    char device_name[32]{};
    char name[32]{};
    TimerStats(const char * name);
    std::string to_csv(bool emitHeader);
};

struct TimerContext{
    private:
    std::chrono::high_resolution_clock::time_point start_time;
    TimerStats& stats;
    public:

    void start();
    void stop();
    TimerContext(const char * name);
    ~TimerContext();
    
    
    template<typename Fnc>
    void with_context(Fnc fnc)
    {
        start();
        fnc();
        stop();
    }
    
    

};

struct GlobalClock{
    private:
    std::vector<TimerStats> stats;
    std::chrono::high_resolution_clock::time_point start_time;
    public:
    static GlobalClock& instance();
    TimerStats& get_or_create(const char * name);

    const decltype(start_time) get_time() const
    {
        return start_time;
    }
    private:
    GlobalClock();
};

