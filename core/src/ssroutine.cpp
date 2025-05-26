#include "log.hpp"
#include "md5.h"
#include "statefile.hpp"
#include "timer.hpp"
#include "utils.hpp"

void brute_routine();
void dict_routine();
void single_node_routine(int argc, char **argv) {
  // Single node routine, no MPI
  TimerStats::set_device_name("single_node");
  debug("Running in single node mode");
  std::string result{};
  if (ARGS.brute_mode()) {
    brute_routine();
  } else {
    dict_routine();
  }
}

void brute_routine() {
  debug("Running brute force routine");
  auto state = StateFile::load(ARGS.save_file);
  size_t addr = 0;
  if (!state.has_value()) {
    debug("No state file found, starting from scratch");
    state = StateFile::create(ARGS.save_file);
  } else {
    addr = state.value()->current_address;
  }

  auto generator = SequenceGenerator{(uint8_t)ARGS.brute_start};
  bool end = false;
  while (!end) {
    TimerContext("BruteTask").with_context([&](TimerStats &s) {
      generator.skip_to(addr);
      auto seq = generator.current();
      uint8_t digest[16]{};
      md5String(const_cast<char *>(seq.c_str()),
    digest);
      char result[33]{};
      md5HexDigest(digest, result);
        if (strncmp(result, ARGS.target_md5, 32) == 0) {
            printf("Found result: %s\n", seq.c_str());
            end = true;
            return ;
        } else {
            addr++;
            if(ARGS.enable_watcher){
                if(addr % ARGS.chunk_size*2 == 0){
                    printf("Current address: %lu, current size: %lu\r", addr, seq.size());
                }
            }
            if (addr % ARGS.chunk_size*10 == 0) {
                state.value()->current_address = addr;
                state.value()->save();
            }
        }
    });
  }
}


void dict_routine() {

  
}