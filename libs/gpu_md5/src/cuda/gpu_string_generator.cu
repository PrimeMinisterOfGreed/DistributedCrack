#include "gpu_string_generator.cuh"
#include "_cdecl"
CDECL 

constexpr int minCharInt = 33;
constexpr int maxCharint = 126;
constexpr int minDigit = 48;
constexpr int maxDigit = 57;
constexpr int minUpperCaseLetter = 65;
constexpr int maxUpperCaseLetter = 90;
constexpr int minLowerCaseLetter = 97;
constexpr int maxLowerCaseLetter = 122;



inline __device__ void grow_swap(char * a, size_t curlen){
    memcpy(a+1,a,curlen);
    a[0]=minCharInt;
}


__device__ void assign_address(GpuStringGenerator* self,size_t address)
{
    self->current_used = false;
    int div = maxCharint - minCharInt;
    int q = address;
    int r = 0;
    int it = 0;
    while (q > 0)
    {
        r = q % div;
        q /= div;
        if (it == self->currentSequenceLength)
        {
            grow_swap(self->current, self->currentSequenceLength);
            self->currentSequenceLength+=1;
        }
        self->current[self->currentSequenceLength - it - 1] = (char)(r + minCharInt);
        it++;
    }

}

__device__ GpuStringGenerator new_generator(int initialSequenceLength)
{
    GpuStringGenerator gen{};
    memset(&gen,0,sizeof(GpuStringGenerator));
    memset(gen.current, minCharInt, 24);
    gen.initialSequenceLength = initialSequenceLength;
    gen.currentSequenceLength = initialSequenceLength;
    return gen;
}


__device__ void next_sequence(GpuStringGenerator* self,char* data){
 if(!self->current_used) {
    memcpy(data,self->current,self->currentSequenceLength);
    self->current_used = true;
    return;
 }
    for (int i = self->currentSequenceLength - 1; i >= 0; i--)
    {
        self->current[i]++;
        if (self->current[i] > maxCharint)
        {
            self->current[i] = minCharInt;
            if (i == 0)
            {
                grow_swap(self->current, self->currentSequenceLength);
                self->currentSequenceLength += 1;
            }
        }
        else
        {
            break;
        }
    }
    memcpy(data,self->current,self->currentSequenceLength);
}

__device__ void destroy_generator(GpuStringGenerator* self){
    
}

END