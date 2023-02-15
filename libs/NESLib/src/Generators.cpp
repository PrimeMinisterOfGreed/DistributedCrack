#include <Generators.hpp>
#include <cmath>
#include "rngs.hpp"

double NegExp(double lambda)
{
    return -log(1 - Random()) / lambda;
}


StreamGenerator *StreamGenerator::_instance;


double StreamGenerator::Random(const BaseStream* stream) const
{
    SelectStream(stream->Stream);
    return ::Random();
}

BaseStream* StreamGenerator::GetStream() 
{
    auto stream = new BaseStream{ _currentStream };
    _currentStream++;
    return stream;
}

BaseStream* BaseRandomVariable::GetStream()
{
    return _stream;
}

StreamGenerator *StreamGenerator::Instance()
{
    if (_instance == nullptr)
        _instance = new StreamGenerator();
    return _instance;
}


double NegExpVariable::GetValue()
{
    return -log(1 - _generator->Random(_stream)) / _lambda;
}

NegExpVariable::NegExpVariable(double lambda, IGenerator *generator) : _lambda(lambda), BaseRandomVariable(generator)
{

}




BaseRandomVariable::BaseRandomVariable(IGenerator *generator) : _generator(generator)
{
    _stream = generator->GetStream();
}

double BaseRandomVariable::operator()()
{
    double result = GetValue();
    OnVariableGenerated.Invoke(result);
    return result;
}

double RandomVariable::GetValue()
{

    return _generator->Random(_stream);
}

RandomVariable::RandomVariable(IGenerator *generator) : BaseRandomVariable(generator)
{

}

double DoubleStageHyperExpVariable::GetValue()
{

    double x = _generator->Random(_stream);
    return _alpha*(1/_u1)* exp(-x/_u1)+_beta*(1/_u2)* exp(-x/_u2);
}

DoubleStageHyperExpVariable::DoubleStageHyperExpVariable(double alpha, double beta, double u1, double u2,
                                                         IGenerator *generator) : BaseRandomVariable(generator),
                                                                                  _alpha(alpha), _u1(u1), _u2(u2),
                                                                                  _beta(beta)
{

}

