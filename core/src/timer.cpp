#include "options.hpp"
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <timer.hpp>

using namespace std::chrono;

char TimerStats::device_name[32] = "unknown";

TimerStats::TimerStats(const char *name) {
  memset(this, 0, sizeof(TimerStats));
  strncpy(this->name, name, sizeof(this->name));
}

std::string TimerStats::to_csv() {
  std::stringbuf buffer;
  std::ostream writer(&buffer);
  auto timestamp = time(NULL);

  writer << TimerStats::device_name << "," << name << "," << ARGS.cpu_threads
         << "," << ARGS.gpu_threads << "," << ARGS.use_gpu << ","
         << ARGS.chunk_size << "," << observation_time << "," << busy_time
         << "," << task_completed << "," << timestamp << "\n";

  return buffer.str();
}

std::string TimerStats::csv_header() {
  const char header[] = "device_name,context_name,cpu_threads,gpu_threads,on_"
                        "gpu,chunk_size,busy_time,"
                        "observation_time,task_completed,time_stamp\n";
  return header;
}

void TimerStats::set_device_name(const char *name) {
  memset(device_name, 0, sizeof(device_name));
  strncpy(device_name, name, std::min(sizeof(device_name), strlen(name)));
}

GlobalClock &GlobalClock::instance() {
  static GlobalClock instance;
  return instance;
}

TimerStats &GlobalClock::get_or_create(const char *name) {
  for (auto &stat : stats) {
    if (strcmp(stat.name, name) == 0) {
      return stat;
    }
  }
  auto stat = TimerStats(name);
  stats.push_back(stat);
  return stats.back();
}

GlobalClock::GlobalClock() {
  start_time = high_resolution_clock::now();
  stats.reserve(32);
}

void TimerContext::start() {
  start_time = std::chrono::high_resolution_clock::now();
}

void TimerContext::stop() {
  auto total = high_resolution_clock::now() - start_time;
  auto busy_time = duration_cast<milliseconds>(total).count();
  stats.busy_time += busy_time;
  stats.observation_time =
      std::chrono::duration_cast<milliseconds>(
          high_resolution_clock::now() - GlobalClock::instance().get_time())
          .count();
}

TimerContext::TimerContext(const char *name)
    : stats(GlobalClock::instance().get_or_create(name)) {}

TimerContext::~TimerContext() { stop(); }

void save_stats(const char *filename) {
  auto file = fopen(filename, "at");
  flockfile(file);
  auto stats = GlobalClock::instance().get_stats();
  for (auto &stat : stats) {
    auto content = stat.to_csv();
    fwrite(content.c_str(), sizeof(char), content.size(), file);
  }
  fflush(file);
  funlockfile(file);
}

void init_stat_file(const char *filename) {
  if (std::filesystem::exists(filename)) {
    return;
  }
  auto file = fopen(filename, "wt");
  fprintf(file, "%s", TimerStats::csv_header().c_str());
  fclose(file);
}
