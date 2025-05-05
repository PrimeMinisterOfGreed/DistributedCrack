#include <timer.hpp>

TimerStats::TimerStats(const char * name)
{
    memset(this, 0, sizeof(TimerStats));
}

GlobalClock& GlobalClock::instance() {
    static GlobalClock instance;
    return instance;
}

TimerStats& GlobalClock::get_or_create(const char * name) {
    for (auto & stat : stats) {
        if (strcmp(stat.name, name) == 0) {
            return stat;
        }
    }
    stats.emplace_back(name);
    return stats.back();
}

GlobalClock::GlobalClock()
{
    start_time = std::chrono::high_resolution_clock::now();
    stats.reserve(32);
}

TimerContext::TimerContext(const char * name): stats(GlobalClock::instance().get_or_create(name))
{
    start_time = std::chrono::high_resolution_clock::now();
}

TimerContext::~TimerContext()
{
    stop();
}

