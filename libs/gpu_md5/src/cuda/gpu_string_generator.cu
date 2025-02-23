#include "gpu_string_generator.cuh"
 
constexpr int minCharInt = 33;
constexpr int maxCharint = 126;
constexpr int minDigit = 48;
constexpr int maxDigit = 57;
constexpr int minUpperCaseLetter = 65;
constexpr int maxUpperCaseLetter = 90;
constexpr int minLowerCaseLetter = 97;
constexpr int maxLowerCaseLetter = 122;


inline __device__ void grow_swap(char * a, size_t curlen){
    char* res = new char[curlen+1]{};
    memcpy(res+1,a,curlen);
    res[0]=minCharInt;
    free(a);
    a = res;
}


__device__ void GpuStringGenerator::assign_address(size_t address)
{
    _current_used = false;
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
            grow_swap(_current, _currentSequenceLength);
            _currentSequenceLength+=1;
        }
        _current[_currentSequenceLength - it - 1] = (char)(r + minCharInt);
        it++;
    }

}

__device__ GpuStringGenerator::GpuStringGenerator(int initialSequenceLength) : _currentAddress(0),
                                                                      _currentSequenceLength(initialSequenceLength)
{
    _current = new char[initialSequenceLength+1]{};
    for (int i = 0; i < _currentSequenceLength; i++)
    {
        _current[i] = minCharInt;
    }
    _current[initialSequenceLength] = '\000';
}


__device__ void GpuStringGenerator::next_sequence(char* data){
 if(!_current_used) {
    memcpy(data,_current,_currentSequenceLength);
    _current_used = true;
    return;
 }
    for (int i = _currentSequenceLength - 1; i >= 0; i--)
    {
        _current[i]++;
        if (_current[i] > maxCharint)
        {
            _current[i] = minCharInt;
            if (i == 0)
            {
                grow_swap(_current, _currentSequenceLength);
                _currentSequenceLength += 1;
            }
        }
        else
        {
            break;
        }
    }
    memcpy(data,_current,_currentSequenceLength);
}

__device__ GpuStringGenerator::~GpuStringGenerator(){
    free(_current);
}