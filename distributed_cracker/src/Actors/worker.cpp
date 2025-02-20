#include "worker.hpp"
#include "MultiThread/functions.hpp"
#include "log_engine.hpp"
#include "md5_gpu.hpp"
#include "options_bag.hpp"
#include <future>
using namespace boost::mpi;
using namespace std;

void operator<<(std::string &val, char *buffer) {
  for (int i = 0; i < val.size(); i++) {
    val[i] = buffer[i];
  }
}

struct operations_control {
private:
  bool gpu_busy = false;
  future<void> gpu_fptr;
  communicator &comm;
  request stoprecv;

public:
  operations_control(communicator &comm) : comm(comm) {
    stoprecv = comm.irecv(0, stop_tag);
  }

  std::vector<std::string> recv_chunks(int chunks, int root) {
    char *buffer = new char[max_elem_alloc * chunks]{};
    size_t sizes[chunks];
    memset(sizes, 0, chunks * sizeof(size_t));
    comm.recv(root, sizes_tag, sizes, chunks);
    comm.recv(root, chunk_tag, buffer, max_elem_alloc * chunks);
    std::vector<std::string> res{};
    res.resize(chunks, "");
    for (size_t i = 0, disp = 0; i < chunks; i++) {
      char str[sizes[i]+1];
      memset(str, 0, sizes[i]+1);
      memcpy(str, &buffer[disp], sizes[i]);
      res[i] = string{str};
      disp += sizes[i];
    }
    delete[] buffer;
    return res;
  }

  void send_result(std::string str) {
    char buffer[result_size]{};
    memcpy(buffer, str.c_str(), str.size());
    comm.send(0, result_tag, buffer, sizeof(buffer));
  }

  void compute(std::vector<std::string> chunk) {
    if(!std::all_of(chunk.begin(),chunk.end(),[](auto&str){return str.size()>0;})){
      dbgln("Malformed chunk, rejecting");
    }
    if (options.use_gpu && !gpu_busy) {
      gpu_busy = true;
      gpu_fptr = async([=, this]() {
        auto res = md5_gpu(chunk);
        for(int i = 0 ; i < chunk.size(); i++){
          if(res[i] == options.target_md5){
            send_result(chunk[i]);
          }
        }
        gpu_busy = false;
      });
    } else {
      auto res = compute_chunk(chunk, options.target_md5, options.num_threads);
      if (res.has_value()) {
        dbgln("Value found {}",res.value());
        send_result(res.value());
      }
    }
  }

  bool is_stop_recv() { return stoprecv.test().has_value(); }

  ~operations_control() { gpu_fptr.wait(); }
};

void worker_routine(communicator &comm) {
  static operations_control controller{comm};
  while (!controller.is_stop_recv()) {
    dbgln("Process{}: waiting for message", comm.rank());
    auto chunk = controller.recv_chunks(options.chunk_size, 0);
    dbgln("Process{}: computing phase", comm.rank());
    if (chunk.size() > 0)
      controller.compute(chunk);
    dbgln("Process{}: computing phase done",comm.rank());
    comm.send(0, available_resp_tag);
  }
}
