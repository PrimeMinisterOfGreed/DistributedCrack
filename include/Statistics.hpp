#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

using namespace boost::accumulators;

using StatAccumulator = accumulator_set<double, features<tag::mean,tag::max,tag::min,tag::variance,tag::sum>>;

class Watcher
{
private:

public:

	void Start();


};