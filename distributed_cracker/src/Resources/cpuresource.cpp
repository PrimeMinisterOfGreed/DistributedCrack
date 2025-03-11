#include "cpuresource.hpp"
#include "MultiThread/functions.hpp"
#include "options_bag.hpp"
#include "string_generator.hpp"

CpuResource::CpuResource(): Resource()
{
    stats.dev_name = "CPU";   
}

bool CpuResource::available() {
    return true;
}

std::vector<ResourceStats> CpuResource::get_stats() {
    return {stats};
}



void CpuResource::compute(ComputeContext &context) {
    std::vector<std::string> chunk{};
    switch (context.type) {
        case ContextTypes::PairSize:{
            auto pair = static_cast<std::pair<size_t, size_t>*>(context.data);
            AssignedSequenceGenerator gen{options.brutestart};
            gen.assign_address(pair->first);
            chunk = gen.generate_chunk(pair->second-pair->first);
        }break;
        case ContextTypes::StringVector:{
            auto ctx = static_cast<std::vector<std::string>*>(context.data);
            chunk = *ctx;
        }break;
    }   
    ClockRegister::tick(this);
    auto res = compute_chunk(chunk, options.target_md5, options.num_threads);
    if(res.has_value()){
        context.result = res.value();
    }
    auto time = ClockRegister::tock(this);
    stats.busy_time += time;
    stats.task_completed += 1;
}


CpuResource cpu_resource{};