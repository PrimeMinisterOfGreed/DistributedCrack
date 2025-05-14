#include "compute.hpp"
#include "mpi/balancer.hpp"
#include "mpi/communicator.hpp"
#include "mpi/mpiprocess.hpp"
#include "mpi/worker.hpp"
#include "options.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cstdint>
#include <log.hpp>
#include <mpi.h>
#include <vector>

void BruteWorker::process() {
  debug("Worker %d started", comm.rank());
  auto process = MpiProcess(comm);
  auto stop = comm.irecv<uint8_t>(MPI_ANY_SOURCE, TERMINATE);
  auto task = comm.irecv_vector<uint64_t>(MPI_ANY_SOURCE, SIZE, 2);
  comm.send_object<uint8_t>(1u, balancer_id, MPITags::TASK);
  process.add_futures(std::move(stop), std::move(task));
  for (;;) {
    trace("Worker %d waiting for promises", comm.rank());
    auto promise = process.wait_any();
    switch (promise->status().MPI_TAG) {
    case TERMINATE:
      debug("Worker %d terminating", comm.rank());
      return;
    case SIZE: {
      trace("Worker %d received task", comm.rank());
      auto task =
          static_cast<BufferedPromise<uint64_t> *>(promise.get())->get_buffer();
      auto result = task[0] + task[1];
      auto ctx = ComputeContext::BruteContext{
          .start = task[0], .end = task[1], .target = ARGS.target_md5};


      TimerContext("BruteTask").with_context([&](TimerStats &s) {
        trace("Worker %d computing range [%lu, %lu]", comm.rank(),
              task[0], task[1]);
              s.task_completed +=  task[1] - task[0];
        auto res = compute({ctx});
        if (res.has_value()) {
          trace("Worker %d found result", comm.rank());
          comm.send_vector<uint8_t>({res.value().begin(), res.value().end()}, 0,
                                    RESULT);
        }
      });

      process.add_future(comm.irecv_vector<uint64_t>(MPI::ANY_SOURCE, SIZE, 2));
      comm.send_object<uint8_t>(1u, balancer_id, MPITags::TASK);
      break;
    }
    default:
      break;
    }
  }
}

void BruteBalancer::process() {
  debug("Balancer %d started", comm.rank());
  auto process = MpiProcess(comm);
  auto stop = comm.irecv<uint8_t>(MPI_ANY_SOURCE, TERMINATE);
  auto worker_request = comm.irecv<uint8_t>(MPI_ANY_SOURCE, TASK);
  process.add_futures(std::move(stop), std::move(worker_request));
  for (;;) {
    trace("Balancer %d waiting for promises", comm.rank());
    auto promise = process.wait_any();
    switch (promise->status().MPI_TAG) {
    case TERMINATE:
      debug("Balancer %d terminating", comm.rank());
      return;
    case TASK: {
      trace("Balancer %d received task", comm.rank());
      if (ranges.empty()) {
        trace("Balancer %d no ranges left requesting tasks", comm.rank());
        // TODO determine an algorithm to decide how many requests catch, for
        // now is chunk_size* size/cluster_degree
        auto numtask = comm.size() / ARGS.cluster_degree;
        debug("Balancer %d requesting %d tasks", comm.rank(), numtask);
        comm.send_object<uint16_t>(numtask, 0, TASK);
        auto ranges = comm.recv_vector<uint64_t>(MPI_ANY_SOURCE, SIZE);
        debug("Balancer %d received ranges with size %d", comm.rank(),
              ranges.size());
        if (ranges.size() % 2 != 0) {
          exception("Invalid number of ranges");
          abort();
        }
        for (int i = 0; i < ranges.size(); i += 2) {
          this->ranges.push_back({ranges[i], ranges[i + 1]});
        }
      }
      auto range = this->ranges.front();
      comm.send_vector<uint64_t>({range.first, range.second},
                                 promise->status().MPI_SOURCE, SIZE);
      this->ranges.erase(this->ranges.begin());
      process.add_future(comm.irecv<uint8_t>(MPI_ANY_SOURCE, TASK));
      break;
    }
    default:
      break;
    }
  }
}

void ChunkWorker::process() {}

void ChunkBalancer::process() {}