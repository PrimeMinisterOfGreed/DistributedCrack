#include "root.hpp"
#include "chunk_loader.hpp"
#include "log_engine.hpp"
#include "options_bag.hpp"
#include "string_generator.hpp"
#include <mpi.h>
using namespace boost::mpi;
using namespace std;

struct GeneratorNode {
  virtual void send(int dest) = 0;
  virtual ~GeneratorNode() = default;
};

struct MpiAwaiter {
  communicator &_comm;
  vector<request> _pending{};
  vector<int> _toprocess{};
  bool _result_available = false;
  char _result[RESULT_SIZE]{};

  MpiAwaiter(communicator &comm) : _comm(comm) {
    _pending.resize(comm.size());
    for (int i = 0; i < comm.size(); i++) {
      if (i != comm.rank())
        _toprocess.push_back(i);
    }
    _pending.push_back(comm.irecv(any_source, RESULT_TAG, _result, RESULT_SIZE));
  }

  bool result_received() { return _result_available; }

  std::string result() const { return _result; }
  void scan() {
    auto itr = _pending.begin();
    while (itr != _pending.end()) {
      if (itr->test().has_value()) {
        if (itr->test().value().tag() == RESULT_TAG) {
          _result_available = true;
        } else {
          _toprocess.push_back(itr->test()->source());
        }
        _pending.erase(itr);
      }
      itr++;
    }
  }

  int get_next() {
    if (_toprocess.size() > 0) {
      auto res = _toprocess.front();
      _toprocess.erase(_toprocess.begin());
      _pending.push_back(_comm.irecv(res, AVAILABLE_RESP_TAG));
      return res;
    }
    return -1;
  }

  void wait_any_request() {
    auto res = boost::mpi::wait_any(_pending.begin(), _pending.end());
    _toprocess.push_back(res.first.source());
    if(res.first.tag() == RESULT_TAG){
      _result_available = true;
    }
    _pending.erase(res.second);
  }

  ~MpiAwaiter() {
    for (int i = 0; i < _comm.size(); i++)
      if (i != _comm.rank()) {
        _comm.send(i, STOP_TAG);
        _comm.send(i,UNLOCK_TAG);
      }
  }
};

struct ChunkGenerator : GeneratorNode {
private:
  ChunkLoader loader{};
  boost::mpi::communicator &comm;

public:
  ChunkGenerator(boost::mpi::communicator &comm) : comm(comm) {}

  bool send_chunks(int chunks, int target) {
    char *buffer = new char[MAX_ELEM_ALLOC * chunks]{};
    size_t sizes[chunks];
    memset(sizes, 0, chunks * sizeof(size_t));
    auto res = loader.get_chunk(chunks, buffer);
    memcpy(sizes, res.data(), chunks * sizeof(size_t));
    comm.send(target, SIZES_TAG, sizes, chunks);
    comm.send(target, CHUNK_TAG, buffer, MAX_ELEM_ALLOC * chunks);
    delete[] buffer;
    return true;
  }

  void send(int dest) override { send_chunks(options.chunk_size, dest); }
};

struct TaskGenerator : GeneratorNode {
  size_t current_address = 0;
  communicator &_comm;
  TaskGenerator(communicator &comm)
      :  _comm(comm) {}

  void send(int dest) override {
    auto s2 = current_address + options.chunk_size;
    size_t send[]{current_address, s2};
    current_address = s2;
    _comm.send(dest, BRUTE_TASK_TAG, send, 2);
  }
};

void root_routine(boost::mpi::communicator &comm) {
  GeneratorNode *node =
      options.use_dictionary()
          ? static_cast<GeneratorNode *>(new ChunkGenerator(comm))
          : new TaskGenerator(comm); // should be the bruter
  MpiAwaiter awaiter{comm};
  dbgln("Sending to {} nodes, dictionary mode:{}", awaiter._toprocess.size(),options.use_dictionary());
  while (!awaiter.result_received()) {
    dbgln("Scanning");
    awaiter.scan();
    auto next = awaiter.get_next();
    dbgln("next is {}",next);
    if (next > -1) {
      dbgln("Sending to {}", next);
      comm.send(next,UNLOCK_TAG);
      node->send(next);
    }
    else{
      dbgln("awaiting message");
      awaiter.wait_any_request();
    }
  }

  println("Password found:{}", awaiter.result());
  delete node;
}
