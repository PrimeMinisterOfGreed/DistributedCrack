#pragma once
#include <bits/types/FILE.h>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <fstream>
#include <string>
#include "CompileMacro.hpp"

template < typename Archive, typename T>
concept Serializable = requires(T t, Archive a) { boost::serialization::serialize<Archive,T>(a); };

template <typename Archive, typename T>
concept Deserializable = requires(T t, Archive a) { deserialize(t);};

template<typename Archive, typename T> concept IOBound =  Serializable<T, Archive> && Deserializable<T, Archive>;

template <typename Archive,Serializable<Archive> Status> class IStatusSaver
{
  public:
    IStatusSaver();

    virtual void Save(Status & status) = 0;
    virtual void Restore() = 0;
};

template <typename Archive, Serializable<Archive> Status> class BaseStatusSaver : public IStatusSaver<Archive,Status>
{
  private:
    std::string _destFile;
  public:
    BaseStatusSaver(std::string file);
    Status &Restore() override;
    void Save(Status & status) override;
};

template <typename Archive, Serializable<Archive> Status>
inline void BaseStatusSaver<Archive, Status>::Save(Status & status) {
    std::fstream fstream{_destFile};
    boost::archive::text_oarchive archive{fstream};
    boost::serialization::serialize(archive, status, VERSION);
}

template <typename Archive, Serializable<Archive> Status>
inline Status &BaseStatusSaver<Archive, Status>::Restore() {
      
}

template <typename Archive, Serializable<Archive> Status>
inline BaseStatusSaver<Archive,Status>::BaseStatusSaver(std::string file)
      : _destFile(file)
{
      
}




