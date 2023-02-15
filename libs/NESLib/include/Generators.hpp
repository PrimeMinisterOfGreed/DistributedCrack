#pragma once

#include "rngs.hpp"
#include <numbers>
#include "EventHandler.hpp"

double NegExp(double lambda);

struct BaseStream
{
	int Stream;
};

class IRandomVariable
{
protected:
	virtual double GetValue() = 0;
public:
    virtual double operator()() = 0;
	virtual BaseStream* GetStream()  = 0;
};


class IGenerator
{
public:
	
	virtual double Random(const BaseStream * stream) const = 0;
	virtual BaseStream* GetStream() = 0;
};

class BaseRandomVariable : public IRandomVariable
{
protected:
	BaseStream* _stream = nullptr;
	IGenerator* _generator = nullptr;
public:
	EventHandler<double> OnVariableGenerated;
	explicit BaseRandomVariable(IGenerator* generator);
	BaseStream* GetStream() override;
	double operator()() override;
};


class StreamGenerator : public IGenerator
{

protected:

	int _currentStream = 0;
	static StreamGenerator* _instance;
public:

	static StreamGenerator* Instance();
	virtual double Random(const BaseStream* stream) const;
	virtual BaseStream* GetStream();
	
};

class RandomVariable : public BaseRandomVariable
{
protected:
	double GetValue() override;

public:
	RandomVariable(IGenerator* generator);

};

class NegExpVariable : public BaseRandomVariable
{


protected:
	double _lambda;
	double GetValue() override;


public:
	

	explicit NegExpVariable(double lambda, IGenerator* generator);
};

class DoubleStageHyperExpVariable : public BaseRandomVariable
{
private:
	double _alpha;
	double _beta;
	double _u1;
	double _u2;
protected:
	double GetValue() override;

public:
	DoubleStageHyperExpVariable(double alpha, double beta, double u1, double u2, IGenerator* generator);
};