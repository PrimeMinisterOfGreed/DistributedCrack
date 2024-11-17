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

class GeneratorException : std::exception
{
  private:
    char *_message;

  public:
    GeneratorException(char *message) : _message(message)
    {
    }
    const char *what() const noexcept override
    {
        return _message;
    }
};

class ISequenceGenerator
{
  public:
    virtual std::string nextSequence() = 0;
    virtual std::vector<std::string>& generateChunk(int num);
};

class IAddressableGenerator
{
    public:
  virtual void AssignAddress(size_t address) = 0;  
};

class SequentialGenerator : public ISequenceGenerator
{
  protected:
    std::string &_current;

  private:
    int _currentSequenceLength;

  public:
    SequentialGenerator(int initialSequenceLength);
    std::string virtual nextSequence();
    int GetCurrentSequenceLength() const
    {
        return _currentSequenceLength;
    }
    
};

class AssignedSequenceGenerator : public SequentialGenerator, public IAddressableGenerator
{
  private:
    uint64_t _currentSequenceIndex;

  public:
    AssignedSequenceGenerator(int initialSequenceLength);
    std::string nextSequence() override;
    void AssignAddress(uint64_t address) override;
    uint64_t GetCurrentIndex()const{return _currentSequenceIndex;}
};

