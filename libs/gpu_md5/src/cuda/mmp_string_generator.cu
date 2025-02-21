#include "mmp_string_generator.cuh"
/*
__device__ const char *mmp_string_generator_gpu::next_sequence()
{
    char *current = new char[_currentSequenceLength];
    memcpy(current, _current, _currentSequenceLength);
    assign_address(_currentAddress++);
    return current;
}


 __global__ void generate_chunk(int num, size_t *sizes,const char ** chunk)
{
    char **result = new const char *[num];
    for (int i = 0; i < num; i++)
    {
        result[i] = next_sequence();
        sizes[i] = _currentSequenceLength;
    }
}
*/

 __device__ void MMPStringGenerator::assign_address(size_t address, const char* generated, size_t* seq_size)
{
    int div = maxCharint - minCharInt;
    int q = address;
    int r = 0;
    int it = 0;
    size_t ssize = *seq_size;
    while (q > 0)
    {
        r = q % div;
        q /= div;
        if (it == ssize)
        {
            char *newCurrent = new char[ssize + 1];
            newCurrent[0] = (char) minCharInt;
            memcpy(newCurrent + 1, _current, ssize);
            ssize++;
        }
        _current[_currentSequenceLength - it - 1] = (char) (r + minCharInt);
        it++;
    }
}

MMPStringGenerator::MMPStringGenerator(int initialSequenceLength) : _currentAddress(0),
                                                                      _currentSequenceLength(initialSequenceLength + 1)
{
    for (int i = 0; i < _currentSequenceLength; i++)
    {
        _current[i] = minCharInt;
    }
    _current[initialSequenceLength] = '\000';
}


