#pragma once
#include "CompileMacro.hpp"
#include <bits/types/FILE.h>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <fstream>
#include <string>
#include "Concepts.hpp"


template <typename Archive, Serializable<Archive> Status> class IStatusSaver
{
  public:
    virtual void Save(Status &status) = 0;
    virtual Status &Restore() = 0;
};

template <Serializable<boost::archive::text_oarchive> Status>
class BaseStatusSaver : public IStatusSaver<boost::archive::text_oarchive, Status>
{
  private:
    std::string _destFile;

  public:
    BaseStatusSaver(std::string file);
    Status &Restore() override;
    void Save(Status &status) override;
};

template <Serializable<boost::archive::text_oarchive> Status> void BaseStatusSaver<Status>::Save(Status &status)
{
    std::ofstream fstream{_destFile};
    boost::archive::text_oarchive archive{fstream};
    archive << status;
}

template <Serializable<boost::archive::text_oarchive> Status> Status &BaseStatusSaver<Status>::Restore()
{
    std::ifstream inStream{_destFile};
    boost::archive::text_iarchive inArchive{inStream};
    Status &status = *new Status();
    inArchive >> status;
    return status;
}

template <Serializable<boost::archive::text_oarchive> Status>
BaseStatusSaver<Status>::BaseStatusSaver(std::string file) : _destFile(file)
{
}
