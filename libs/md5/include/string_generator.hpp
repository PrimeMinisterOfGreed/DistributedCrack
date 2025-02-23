#pragma once
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>



constexpr int minCharInt = 33;
constexpr int maxCharint = 126;
constexpr int minDigit = 48;
constexpr int maxDigit = 57;
constexpr int minUpperCaseLetter = 65;
constexpr int maxUpperCaseLetter = 90;
constexpr int minLowerCaseLetter = 97;
constexpr int maxLowerCaseLetter = 122;



class ISequenceGenerator
{
  public:
    virtual std::string next_sequence() = 0;
    virtual std::vector<std::string> generate_chunk(int num);
    virtual void generate_chunk(char*buffer,size_t*sizes, int num);
};

class IAddressableGenerator
{
    public:
  virtual void assign_address(size_t address) = 0;  
};

class SequentialGenerator : public ISequenceGenerator
{
  protected:
    std::string _current;

  private:
    int _currentSequenceLength;

  public:
    SequentialGenerator(int initialSequenceLength);
    std::string virtual next_sequence();
    int get_current_length() const
    {
        return _currentSequenceLength;
    }
    
};

class AssignedSequenceGenerator : public SequentialGenerator, public IAddressableGenerator
{
  private:
    uint64_t _currentSequenceIndex;
    bool _currentUsed = false;
  public:
    AssignedSequenceGenerator(int initialSequenceLength);
    std::string next_sequence() override;
    void assign_address(uint64_t address) override;
    uint64_t GetCurrentIndex()const{return _currentSequenceIndex;}
};

