#include "log.hpp"
#include "mpi/balancer.hpp"
#include "mpi/communicator.hpp"
#include "mpi/generator.hpp"
#include "mpi/worker.hpp"
#include "options.hpp"
#include "timer.hpp"
#include <filesystem>
void mpi_routine(int argc, char **argv);

int main(int argc, char **argv) {
  
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
    return 1;
  }
  const char *config_file = argv[1];
  trace("Loading config file: %s\n", config_file);
  auto opt = options::load_from_file(config_file);
  if (opt.is_error()) {
    fprintf(stderr, "Error loading config file: %s\n", opt.error);
    return 1;
  }
  trace("hell", 1);
  ARGS = opt.unwrap();
  if (ARGS.use_mpi) {
    mpi_routine(argc, argv);
  }
}

void root_routine(Communicator &comm);
void worker_routine(Communicator &comm);
void mpi_routine(int argc, char **argv) {
  auto universe = MpiContext{argc, argv};
  auto world = universe.world();
  if (world.rank() == 0) {
    // root node
    debug("Starting root node");
    init_stat_file("stats.csv");
    root_routine(world);
  } else {
    // worker node
    debug("Starting worker node");
    worker_routine(world);
  }
  save_stats("stats.csv");
}

enum NodeType {
  MASTER,
  WORKER,
  BALANCER,
};

void node_assign(Communicator &comm) {
  auto cf = ARGS.cluster_degree;
  // if cluster degree is 0, all nodes are workers
  if (cf == 0) {
    for (int i = 1; i < comm.size(); i++) {
      trace("Assigning node %d , role %d", i, WORKER);
      comm.send_object(WORKER, i, 0);
      comm.send_object(MASTER, i, 1); // set reference to master
    }
    return;
  }
  if (ARGS.cluster_degree >= (comm.size() / 2)) {
    error("Cluster degree is greater than or equal to the number of nodes, "
          "decrementing to size / 2");
    ARGS.cluster_degree = comm.size() / 2;
  }
  for (int i = 1; i < ARGS.cluster_degree + 1; i++) {
    trace("Assigning node %d , role %d", i, BALANCER);
    comm.send_object(BALANCER, i, 0);
  }

  int sel_balancer = 0;
  // this is a naive way to balance the tree, still it can work
  for (int i = ARGS.cluster_degree + 1; i < comm.size(); i++) {
    trace("Assigning node %d , role %d", i, WORKER);
    comm.send_object(WORKER, i, 0);
    comm.send_object(sel_balancer + 1, i, 1);
    sel_balancer = (sel_balancer + 1) % cf;
  }
}

void root_routine(Communicator &comm) {
  node_assign(comm);
  TimerStats::set_device_name("root");
  TimerContext("wallclock").with_context([&](TimerStats&s) {
    s.task_completed+=1;
    auto result = [&comm] {
      if (!ARGS.brute_mode()) {
        return ChunkedGenerator{comm}.process();
      } else {
        return BruteGenerator{comm}.process();
      }
    }();
    printf("Result: %s\n", result.value().c_str());
  });
}

void worker_routine(Communicator &comm) {
  auto node_type = comm.recv_object<NodeType>(0, 0);
  char node_name[32]{};
  if (node_type == WORKER) {

    sprintf(node_name, "worker %d", comm.rank());
    TimerStats::set_device_name(node_name);
    auto balancer_id = comm.recv_object<int>(0, 1);
    if (ARGS.brute_mode()) {
      trace("Worker %d, balancer %d mode brute", comm.rank(), balancer_id);
      BruteWorker{comm, balancer_id}.process();
    } else {
      trace("Worker %d, balancer %d mode chunk", comm.rank(), balancer_id);
      ChunkWorker{comm, balancer_id}.process();
    }
  } else if (node_type == BALANCER) {
    sprintf(node_name, "balancer %d", comm.rank());
    TimerStats::set_device_name(node_name);
    if (ARGS.brute_mode()) {
      trace("Balancer %d mode brute", comm.rank());
      BruteBalancer{comm}.process();
    } else {
      trace("Balancer %d mode chunk", comm.rank());
      ChunkBalancer{comm}.process();
    }
  }
}
