#include "root.hpp"
#include "chunk_loader.hpp"
#include "log_engine.hpp"
#include "string_generator.hpp"
#include "options_bag.hpp"
#include <mpi.h>
using namespace boost::mpi;
using namespace std;

struct chunk_sender{
  private:
  vector<request> pending{};
  request result_req;
  vector<bool> first{};
  char _result[result_size]{};
  ChunkLoader loader{};
  boost::mpi::communicator & comm;
  public:
  chunk_sender(boost::mpi::communicator&comm): comm(comm){
    pending.resize(comm.size());
    first.resize(comm.size(),false);
    result_req = comm.irecv(any_source,result_tag,_result,result_size);
  }

  bool send_chunks(int chunks, int target) {
    if (first[target] && !pending[target].test().has_value()) {
      return false;
    }
    char *buffer = new char[max_elem_alloc * chunks]{};
    size_t sizes[chunks];
    memset(sizes, 0, chunks * sizeof(size_t));
    auto res = loader.get_chunk(chunks, buffer);
    memcpy(sizes, res.data(),chunks * sizeof(size_t));
    comm.send(target, sizes_tag, sizes, chunks);
    comm.send(target, chunk_tag, buffer, max_elem_alloc * chunks);
    pending[target] = comm.irecv(target, available_resp_tag);
    first[target] = true;
    delete[] buffer;
    return true;
  }

  bool result_received() { return result_req.test().has_value(); }
  int wait_any_pending() {
      if (!std::all_of(pending.begin(), pending.end(), [](request &elem) {
            return elem.test().is_initialized();
          })) {
        return -1;
      }

    for (auto &pend : pending) {
      if (pend.test().has_value()) {
        return pend.test().get().source();
      }
    }

    auto towait = pending;
    towait.push_back(result_req);
    auto res = wait_any(pending.begin(), pending.end());
    return res.first.source();
  }

  int get_next_process() {
    static bool allinit = false;
    if (!allinit)
      for (int i = 1; i < first.size(); i++) {
        if (!first[i])
          return i;
      }
    allinit = true;
    return wait_any_pending();
  }
  const char* result() const{return _result;}
};

void root_routine(boost::mpi::communicator &comm) {
  chunk_sender sender{comm};
  dbgln("Sending all processes a chunk");

  while (!sender.result_received()) {
    auto next = sender.get_next_process();
    if (next != -1) {
      dbgln("Sending chunk to {}", next);
      sender.send_chunks(options.chunk_size, next);
    }
    else{
      for(int i = 1; i < comm.size(); i++){
        sender.send_chunks(options.chunk_size, i);
      }
    }
  }
  for (int i = 1; i < comm.size(); i++)
    comm.send(i, stop_tag);
  println("Password found:{}", sender.result());
}
