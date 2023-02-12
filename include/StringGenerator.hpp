#pragma once
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

class SequentialGenerator
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
    virtual std::vector<std::string> &generateChunk(int num);
};

class AssignedSequenceGenerator : public SequentialGenerator
{
  private:
    uint64_t _currentSequenceIndex;

  public:
    AssignedSequenceGenerator(int initialSequenceLength);
    std::string nextSequence() override;
    void AssignAddress(uint64_t address);
};

class MultiThreadStringGenerator : public AssignedSequenceGenerator
{
  private:
    std::mutex& _guard = *new std::mutex();
  public:
    MultiThreadStringGenerator(int initialSequenceLength);
    std::vector<std::string> & SafeGenerateChunk(int num);
};