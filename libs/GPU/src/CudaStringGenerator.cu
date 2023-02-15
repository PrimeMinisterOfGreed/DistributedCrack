#include "CudaStringGenerator.cuh"

__host__ __device__ const char *CudaStringGenerator::nextSequence()
{
    char *current = new char[_currentSequenceLength];
    memcpy(current, _current, _currentSequenceLength);
    assignAddress(_currentAddress++);
    return current;
}

__host__ __device__ const char **CudaStringGenerator::generateChunk(int num, size_t *sizes)
{
    const char **result = new const char *[num];
    for (int i = 0; i < num; i++)
    {
        result[i] = nextSequence();
        sizes[i] = _currentSequenceLength;
    }
    return result;
}

__host__ __device__ void CudaStringGenerator::assignAddress(size_t address)
{
    int div = maxCharint - minCharInt;
    int q = address;
    int r = 0;
    int it = 0;
    while (q > 0)
    {
        r = q % div;
        q /= div;
        if (it == _currentSequenceLength)
        {
            char *newCurrent = new char[_currentSequenceLength + 1];
            newCurrent[0] = (char) minCharInt;
            memcpy(newCurrent + 1, _current, _currentSequenceLength);
            _currentSequenceLength++;
        }
        _current[_currentSequenceLength - it - 1] = (char) (r + minCharInt);
        it++;
    }
}

CudaStringGenerator::CudaStringGenerator(int initialSequenceLength) : _currentAddress(0),
                                                                      _currentSequenceLength(initialSequenceLength + 1)
{
    for (int i = 0; i < _currentSequenceLength; i++)
    {
        _current[i] = minCharInt;
    }
    _current[initialSequenceLength] = '\000';
}


