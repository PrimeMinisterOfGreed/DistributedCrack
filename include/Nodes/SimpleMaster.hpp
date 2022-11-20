#include "Nodes/Node.hpp"
#include <boost/mpi.hpp>

class SimpleMaster : public Node
{
  private:
    boost::mpi::communicator _comm;
    std::vector<Statistics> _collectedStats{};
    std::string _target;
    std::vector<boost::mpi::request>& _requests= *new std::vector<boost::mpi::request>{};
    std::string _result = "";
    void DeleteRequest(boost::mpi::request* request);
  public:
    SimpleMaster(boost::mpi::communicator comm, std::string target);
    virtual void OnEndRoutine() override;
    virtual void Initialize() override;
    virtual void Routine() override;
    virtual void OnBeginRoutine() override;
};