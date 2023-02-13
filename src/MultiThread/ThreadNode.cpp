#include "MultiThread/ThreadNode.hpp"

ThreadNode::ThreadNode(ISequenceGenerator &generator) : Node(), _generator(generator)
{
}

void ThreadNode::Initialize()
{
    Node::Initialize();
}

void ThreadNode::Routine()
{
    Node::Routine();
    
}

void ThreadNode::OnBeginRoutine()
{
    Node::OnBeginRoutine();
}

void ThreadNode::OnEndRoutine()
{
    Node::OnEndRoutine();
}
