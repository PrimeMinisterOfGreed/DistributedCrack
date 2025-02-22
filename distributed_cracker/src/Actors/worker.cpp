#include "worker.hpp"
#include "MultiThread/functions.hpp"
#include "log_engine.hpp"
#include "md5_gpu.hpp"
#include "options_bag.hpp"
#include "string_generator.hpp"
#include <future>
using namespace boost::mpi;
using namespace std;

// Mpi Node

void operator<<(std::string &val, char *buffer) {
  for (int i = 0; i < val.size(); i++) {
    val[i] = buffer[i];
  }
}


struct WorkerNode{
  private:
  public:
  virtual void recv_phase() = 0;
  virtual void compute_phase() = 0;
  static void send_result(std::string str, communicator&comm) {
    char buffer[RESULT_SIZE]{};
    memcpy(buffer, str.c_str(), str.size());
    comm.send(0, RESULT_TAG, buffer, sizeof(buffer));
  }
  virtual ~WorkerNode() = default;
};

struct ChunkNode : WorkerNode {
  communicator &comm;

  ChunkNode(communicator &comm) : comm(comm) {
  }

  std::vector<std::string> recv_chunks(int chunks) {
    char *buffer = new char[MAX_ELEM_ALLOC * chunks]{};
    size_t sizes[chunks];
    memset(sizes, 0, chunks * sizeof(size_t));
    comm.recv(any_source, SIZES_TAG, sizes, chunks);
    comm.recv(any_source, CHUNK_TAG, buffer, MAX_ELEM_ALLOC * chunks);
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





  void compute(std::vector<std::string> chunk) {
    if (options.use_gpu) {
        auto res = md5_gpu(chunk);
        for(int i = 0 ; i < chunk.size(); i++){
          if(res[i] == options.target_md5){
            send_result(chunk[i], comm);
          }
        }
    } else {
      auto res = compute_chunk(chunk, options.target_md5, options.num_threads);
      if (res.has_value()) {
        dbgln("Value found {}",res.value());
        send_result(res.value(), comm);
      }
    }
  }

  ~ChunkNode() { 
    //gpu_fptr.wait();
     }

     //interface
     std::vector<std::string> chunk;
     void recv_phase() override{
      chunk = recv_chunks(options.chunk_size);
     }

     void compute_phase() override{
      compute(chunk);
     }

};



struct TaskNode : WorkerNode{
  communicator& comm;
  AssignedSequenceGenerator generator;
  TaskNode(communicator&comm):comm(comm),generator(options.brutestart){}

  std::pair<size_t, size_t> recv_task(){ 
    size_t sizes[4]{};
    comm.recv(any_source,BRUTE_TASK_TAG,sizes,2);
    return {sizes[0],sizes[1]};
  }

  void compute_task(std::pair<size_t, size_t> task){
    auto res = optional<std::string>{};
    res.reset(); //just in case
    if(options.use_gpu){
      res= md5_bruter(task.first, task.second, options.target_md5);
    }
    else{
      generator.assign_address(task.first);
      auto chunk = generator.generate_chunk(task.second-task.first);
      res = compute_chunk(chunk, options.target_md5, options.num_threads);
    }

    if(res.has_value()){
      send_result(res.value(), comm);
    }
  }

  //interface
  std::pair<size_t, size_t> task;
  void compute_phase() override{
    compute_task(task);
  }

  void recv_phase() override{
    task = recv_task();
  }
};



void worker_routine(communicator &comm) {
  request stop_request;
  request unlock_request;
  stop_request = comm.irecv(any_source, STOP_TAG); //wait for someone to say stop
  auto node = std::unique_ptr<WorkerNode>(options.use_dictionary()
               ? static_cast<WorkerNode*>(new ChunkNode(comm))
               : new TaskNode(comm));
  std::vector<request> reqs{stop_request, unlock_request};
  while (!stop_request.test().has_value()) {
    comm.recv(any_source, UNLOCK_TAG);
    if(stop_request.test().has_value()){
      dbgln("shutting down");
      return;
    }
    dbgln("Process{}: waiting for message", comm.rank());
    node->recv_phase();
    dbgln("Process{}: computing phase", comm.rank());
    node->compute_phase();
    dbgln("Process{}: computing phase done", comm.rank());
    comm.isend(0, AVAILABLE_RESP_TAG);
  }
}

// Single Worker
