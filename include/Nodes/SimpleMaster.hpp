#include "Nodes/Node.hpp"
#include <boost/mpi.hpp>

class SimpleMaster : public MPINode
{
  private:
    boost::mpi::communicator _comm;
    std::vector<Statistics> _collectedStats{};
    std::string _result;
    virtual void OnEndRoutine() override;
    virtual void Initialize() override;
    virtual void Routine() override;
    virtual void OnBeginRoutine() override;
  public:
    SimpleMaster(boost::mpi::communicator comm, std::string target) : MPINode(comm, target) {};
    void Report();
};